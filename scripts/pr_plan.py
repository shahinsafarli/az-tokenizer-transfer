"""Local PR-plan lookup, materialization, hash audit and Git scope inspection.

This tool never logs in, changes Git identity, commits, pushes, opens or merges a PR.
The AI follows DO_PR.md to perform those separately under the requested PR.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def normalize_id(value):
    match=re.fullmatch(r'(?:PR)?0*(\d+)',str(value).strip(),re.IGNORECASE)
    check(match is not None,'Use a PR id such as PR05 or 5')
    number=int(match.group(1))
    check(0 <= number <= 19,'This plan has bootstrap 0 and PR01–PR19')
    return f'PR{number:02}'


def plan_root():
    # Script lives under scripts/, both in the handoff bundle and in the repo.
    return Path(__file__).resolve().parent.parent


def load_context():
    root=plan_root()
    return root,read_json(root/'.github/pr-plan.yml'),read_json(root/'.github/import-manifest.json')


def selected(manifest, pid):
    n=int(pid[2:]); answer=[]
    for entry in manifest['files']:
        if entry.get('pr')==n:
            answer.append((entry,None))
        for part in entry.get('parts',[]):
            if part['pr']==n: answer.append((entry,part))
    return answer


def safe_path(root, relative):
    path=(root/relative).resolve()
    check(path.is_relative_to(root.resolve()),'Path escapes target directory: '+relative)
    return path


def read_origin(input_dir, origin):
    if '!' in origin:
        archive,member=origin.split('!',1)
        with zipfile.ZipFile(safe_path(input_dir,archive)) as z:
            return z.read(member)
    return safe_path(input_dir,origin).read_bytes()


def retrieve(input_dir, entry):
    # Only named, hashed archive members in the generated manifest are read.
    origin=entry['origins'][0]
    raw=read_origin(input_dir,origin)
    check(hashlib.sha256(raw).hexdigest()==entry['sha256'],'Input hash mismatch: '+entry['path'])
    return raw


def materialize(manifest,pid,input_dir,output):
    output.mkdir(parents=True,exist_ok=True)
    prepared=[]
    for entry,part in selected(manifest,pid):
        raw=retrieve(input_dir,entry)
        path=safe_path(output,entry['path'])
        if part:
            lines=raw.splitlines(keepends=True)
            before=b''.join(lines[:part['start']-1])
            after=b''.join(lines[:part['end']])
            current=path.read_bytes() if path.exists() else b''
            check(current in (before,after),f'{entry["path"]}: expected the exact prior historical-document prefix')
            raw=after
        elif path.exists():
            check(path.read_bytes()==raw,'Refusing to overwrite a different existing file: '+entry['path'])
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw);prepared.append(entry['path'])
    return prepared


def audit(manifest,repo,pid=None,phase=None):
    entries=selected(manifest,pid) if pid else [(e,None) for e in manifest['files'] if phase is None or e['phase']==phase]
    failures=[]
    for entry,part in entries:
        path=safe_path(repo,entry['path'])
        if not path.is_file(): failures.append('missing: '+entry['path']);continue
        raw=path.read_bytes()
        if part and part['end']!=entry.get('physical_lines'):
            # PR06's incomplete historical document has its own prefix hash.
            expected=part['prefix_sha256']
        else: expected=entry['sha256']
        if hashlib.sha256(raw).hexdigest()!=expected: failures.append('hash changed: '+entry['path'])
    check(not failures,'Snapshot audit failed:\n'+'\n'.join(failures))
    if not pid and phase is None:
        runs=list((repo/'results/runs').glob('*.json'))
        logs=list((repo/'results/logs').glob('*.log'))
        check(len(runs)==164 and len(logs)==164,'Expected 164 final runs and 164 logs')
        keys=[]
        for path in runs:
            record=read_json(path)
            check(record.get('run_result_schema_version')==4,'Mixed run schema: '+path.name)
            keys.append((record['base'],record['condition'],record['train_size'],record['seed']))
            check(len(list((repo/'results/logs').glob(path.stem+'__*.log')))==1,'Unpaired log: '+path.name)
        check(len(set(keys))==164,'Duplicate final run identity')
        state=read_json(repo/'results/launcher_state.json')
        check(state['queue_count']==164 and len(set(state['completed']))==164,'Launcher count mismatch')
    return len(entries)


def git(repo,*args):
    result=subprocess.run(['git','-C',str(repo),*args],capture_output=True,check=True)
    return result.stdout


def changed_paths(repo,base):
    # NUL separation preserves spaces; inspect commits, index, worktree and untracked files.
    commands=[('diff','--name-only','-z',f'{base}...HEAD'),('diff','--name-only','-z'),
              ('diff','--cached','--name-only','-z'),('ls-files','--others','--exclude-standard','-z')]
    found=set()
    for args in commands:
        found.update(x.decode('utf8') for x in git(repo,*args).split(b'\0') if x)
    return found


def scope(pr,repo,base):
    changed=changed_paths(repo,base)
    stray=sorted(p for p in changed if not any(fnmatch.fnmatchcase(p,pattern) for pattern in pr['paths']))
    check(not stray,'Out-of-scope paths:\n'+'\n'.join(stray))
    forbidden=[]
    for p in changed:
        low=p.lower()
        if any(segment in low.split('/') for segment in ['.claude','_repo_setup','artifacts','.venv']) or re.search(r'(?:\.pat_|\.device_codes|\.env$|\.(?:safetensors|pt|bin|tgz|rar)$)',low):
            forbidden.append(p)
    check(not forbidden,'Forbidden repository content: '+', '.join(forbidden))
    too_big=[p for p in changed if safe_path(repo,p).is_file() and safe_path(repo,p).stat().st_size>5_000_000]
    check(not too_big,'Files above 5 MB: '+', '.join(too_big))
    branch=git(repo,'branch','--show-current').decode().strip()
    check(branch==pr['branch'],f'Expected branch {pr["branch"]}, found {branch}')
    return sorted(changed)


def validate_registry(plan,manifest):
    check(plan['meta']['plan_id']=='balanced-edited-v4-4-commits','Wrong plan version')
    check(manifest['plan_id']==plan['meta']['plan_id'],'Registry/manifest version mismatch')
    check(plan['meta']['source_archive_sha256']==manifest['input_archives'][plan['meta']['source_archive']]['sha256'],'Code archive hash/version mismatch')
    prs=plan['prs'];ids=[p['id'] for p in prs]
    check(ids==[f'PR{i:02}' for i in range(1,20)],'Incomplete PR registry')
    counts=Counter(p['owner'] for p in prs);counts['SS']+=1
    check(set(counts.values())=={4},'Expected four planned main commits each, including SS bootstrap')
    check(plan['meta']['planned_pr_count']==19 and plan['meta']['main_commits_per_member']==4,'Commit accounting metadata mismatch')
    check(manifest['plan_id']==plan['meta']['plan_id'],'Manifest belongs to a different plan')
    by={p['id']:p for p in prs}
    def visit(pid,active,done):
        check(pid not in active,'Dependency cycle at '+pid)
        if pid in done:return
        active.add(pid)
        for dep in by[pid]['depends_on']:
            check(dep in by,'Unknown dependency '+dep);visit(dep,active,done)
        active.remove(pid);done.add(pid)
    for pid in ids:visit(pid,set(),set())
    check(len({e['path'] for e in manifest['files']})==len(manifest['files']),'Duplicate manifest destination')
    for entry in manifest['files']:
        for n in ([entry['pr']] if entry.get('pr') is not None else [p['pr'] for p in entry['parts']]):
            if n==0:continue
            check(entry['path'] in by[f'PR{n:02}']['paths'],'Import outside registered exact paths: '+entry['path'])
    for p in prs:
        check(p['owner'] not in p['reviewers'],'Self-review assignment')
        check(len(p['commits'])==1,'Use one final authored commit per planned PR')
        check(p['validate'] and p['review_criteria'],'Incomplete execution contract: '+p['id'])
        if int(p['id'][2:])>=15:
            for entry in manifest['files']:
                if entry['path'].startswith('src/') and entry['path'] in p['paths'] and entry.get('owner')!=p['owner']:
                    check(entry['owner'] in p['reviewers'],'Missing existing source-owner review: '+p['id']+' '+entry['path'])
    return counts



def check_inputs(manifest, input_dir):
    for name, info in manifest['input_archives'].items():
        path=safe_path(input_dir,name)
        check(path.is_file(),'Missing required input: '+name)
        check(hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256'],'Wrong input archive/file: '+name)
    return len(manifest['input_archives'])


def audit_test_access(manifest, repo, stage):
    # Import-milestone verification only; no project imports or model execution.
    by={e['path']:e for e in manifest['files']}
    source_paths=['src/training/finetune.py','src/training/run_grid.py',
                  'src/training/orchestrate.py','src/training/phases.py',
                  'src/analysis/errors.py','run_all.sh','scripts/vastai_run_full.sh',
                  'configs/CONFIG_HASHES.lock','configs/FROZEN.md',
                  'configs/experiment.yaml','configs/pilot_t4.yaml','docs/TEST_AUDIT.md']
    for rel in source_paths:
        check(hashlib.sha256(safe_path(repo,rel).read_bytes()).hexdigest()==by[rel]['sha256'],
              'Test-access baseline drift: '+rel)
    lock=read_json(repo/'configs/CONFIG_HASHES.lock')
    for rel in ['configs/FROZEN.md','configs/experiment.yaml','configs/pilot_t4.yaml']:
        raw=safe_path(repo,rel).read_bytes().replace(b'\r\n',b'\n')
        check(hashlib.sha256(raw).hexdigest()==lock[rel],'Config lock mismatch: '+rel)
    result={'stage':stage,'source_hashes_and_config_lock':'PASS',
            'python_default':'val; test requires explicit opt-in',
            'full_shell_wrapper_default':'test with opt-in flags supplied by wrapper',
            'model_evaluation_performed':False}
    if stage=='evidence':
        prefix='evidence/source_snapshot/'
        ledger=safe_path(repo,prefix+'results/test_evaluation_ledger.jsonl')
        check(ledger.read_bytes()==b'','Edited snapshot ledger must match its empty supplied bytes')
        records=[p for p in manifest['source_update']['changed'] if p.endswith('.json') and '/runs/' in p]
        check(len(records)==22,'Expected 22 reduced historical run records')
        for rel in records:
            path=safe_path(repo,prefix+rel)
            check(hashlib.sha256(path.read_bytes()).hexdigest()==by[prefix+rel]['sha256'],'Edited historical record drift: '+rel)
            record=read_json(path)
            check(not {'test','test_predictions','test_gold'} & record.keys(),'Removed test fields restored: '+rel)
        asrun=safe_path(repo,'results/test_evaluation_ledger.jsonl')
        check(hashlib.sha256(asrun.read_bytes()).hexdigest()==by['results/test_evaluation_ledger.jsonl']['sha256'],'As-run ledger drift')
        count=sum(bool(line.strip()) for line in asrun.read_text().splitlines())
        check(count==164,'Expected 164 entries in the separate as-run ledger')
        result.update(edited_snapshot_ledger_entries=0,as_run_ledger_entries=count,
                      historical_reduced_records=22,lifetime_test_access='Not inferred from an empty edited ledger')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('verify-plan')
    inp=sub.add_parser('check-inputs');inp.add_argument('--input-dir',type=Path,required=True)
    access=sub.add_parser('audit-test-access');access.add_argument('--repo',type=Path,required=True);access.add_argument('--stage',choices=['source','evidence'],required=True)
    show=sub.add_parser('show');show.add_argument('pr')
    material=sub.add_parser('materialize');material.add_argument('pr');material.add_argument('--input-dir',type=Path,required=True);material.add_argument('--output',type=Path,required=True)
    aud=sub.add_parser('audit');aud.add_argument('--repo',type=Path,required=True);aud.add_argument('--pr');aud.add_argument('--phase',choices=['source','evidence'])
    sc=sub.add_parser('scope');sc.add_argument('pr');sc.add_argument('--repo',type=Path,default=Path.cwd());sc.add_argument('--base',default='origin/main')
    args=parser.parse_args();root,plan,manifest=load_context()
    if args.command=='check-inputs':
        print(json.dumps({'verified_inputs':check_inputs(manifest,args.input_dir)},indent=2));return
    if args.command=='audit-test-access':
        print(json.dumps(audit_test_access(manifest,args.repo,args.stage),indent=2));return
    if args.command=='verify-plan':
        counts=validate_registry(plan,manifest);print(json.dumps({'plan_id':plan['meta']['plan_id'],'prs':len(plan['prs']),'planned_main_commits':counts,'files':len(manifest['files'])},indent=2));return
    if args.command=='audit':
        n=audit(manifest,args.repo,normalize_id(args.pr) if args.pr else None,args.phase);print(f'PASS: {n} assigned import entries match their frozen hashes.');return
    pid=normalize_id(args.pr)
    pr=next((p for p in plan['prs'] if p['id']==pid),plan['bootstrap'] if pid=='PR00' else None)
    check(pr is not None,'Unknown PR')
    if args.command=='show':print(json.dumps(pr,ensure_ascii=False,indent=2))
    elif args.command=='materialize':
        check_inputs(manifest,args.input_dir)
        paths=materialize(manifest,pid,args.input_dir,args.output)
        print(json.dumps({'prepared_import_files':len(paths),'paths':paths,'remaining_work':pr['objective'],'status':'Files prepared locally; no Git operation performed.'},ensure_ascii=False,indent=2))
    elif args.command=='scope':print(json.dumps({'in_scope':scope(pr,args.repo,args.base)},indent=2))


if __name__=='__main__':
    main()
