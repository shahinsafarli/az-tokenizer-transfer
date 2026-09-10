"""Stage-aware Git hygiene. Standard library only; never runs model experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.parse
import urllib.request

PLAN_ID = 'balanced-edited-v4-4-commits'
REPOSITORY = 'shahinsafarli/az-tokenizer-transfer'
INITIAL_REGISTRY_SHA256 = '40fb08163271a9e59c5247bfb6793b540b4a26673007d4b053f3011f2cb34bf3'
# Generated verbatim from the supplied registry's first four execution cards.
# PR04 installs the full registry. Until then no absent registry is required.
EARLY = {'PR01': {'id': 'PR01',
          'title': 'PR01 \xb7 ci(repo): ci foundation',
          'branch': 'ci/pr01-ej-ci-foundation',
          'paths': ['.github/scripts/pr_hygiene.py',
                    '.github/workflows/ci.yml',
                    'LICENSE',
                    'NOTICE',
                    'requirements-ci.txt'],
          'depends_on': [],
          'commits': ['ci(repo): ci foundation'],
          'login': 'Emil-Jafarov-06'},
 'PR02': {'id': 'PR02',
          'title': 'PR02 \xb7 docs(repo): governance',
          'branch': 'docs/pr02-fa-governance',
          'paths': ['.github/CODEOWNERS',
                    '.github/ISSUE_TEMPLATE/bug.yml',
                    '.github/ISSUE_TEMPLATE/config.yml',
                    '.github/ISSUE_TEMPLATE/finding.yml',
                    '.github/labels.yml',
                    '.github/pull_request_template.md',
                    'CONTRIBUTING.md'],
          'depends_on': [],
          'commits': ['docs(repo): governance'],
          'login': 'FatimaAlibabayeva'},
 'PR03': {'id': 'PR03',
          'title': 'PR03 \xb7 build(repo): local verification',
          'branch': 'build/pr03-fb-local-verification',
          'paths': ['Makefile', 'ruff.toml', 'scripts/verify_repo.py'],
          'depends_on': ['PR01'],
          'commits': ['build(repo): local verification'],
          'login': 'Fidan6557'},
 'PR04': {'id': 'PR04',
          'title': 'PR04 \xb7 chore(repo): registry and writing skeletons',
          'branch': 'chore/pr04-sa-registry-and-writing-skeletons',
          'paths': ['.github/import-manifest.json',
                    '.github/pr-plan.yml',
                    'DO_PR.md',
                    'GIT_WORKFLOW_AND_CONTRIBUTIONS.md',
                    'docs/CODEBASE_UPDATE.md',
                    'evidence/README_REPOSITORY_PATHS.md',
                    'presentation/README.md',
                    'presentation/beamerthemeAIA.sty',
                    'presentation/presentation.tex',
                    'presentation/sections/ej.tex',
                    'presentation/sections/fa.tex',
                    'presentation/sections/fb.tex',
                    'presentation/sections/sa.tex',
                    'presentation/sections/ss.tex',
                    'report/README.md',
                    'report/claims/ej.yaml',
                    'report/claims/fa.yaml',
                    'report/claims/fb.yaml',
                    'report/claims/sa.yaml',
                    'report/claims/ss.yaml',
                    'report/latexmkrc',
                    'report/references/ej.bib',
                    'report/references/fa.bib',
                    'report/references/fb.bib',
                    'report/references/sa.bib',
                    'report/references/ss.bib',
                    'report/refs.bib',
                    'report/report.tex',
                    'report/sections/00_abstract.tex',
                    'report/sections/01_introduction.tex',
                    'report/sections/02_related_work.tex',
                    'report/sections/03_method.tex',
                    'report/sections/04_setup_data.tex',
                    'report/sections/05_setup_training.tex',
                    'report/sections/06_results.tex',
                    'report/sections/07_discussion_limitations.tex',
                    'report/sections/08_conclusion.tex',
                    'report/sections/09_tools_ack.tex',
                    'scripts/pr_plan.py'],
          'depends_on': ['PR01', 'PR02', 'PR03'],
          'commits': ['chore(repo): registry and writing skeletons'],
          'login': 'SuleymanAllahverdiyev'}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(*args):
    return subprocess.check_output(['git', *args])


def checked_sha(value):
    require(re.fullmatch(r'[0-9a-f]{40}', value or '') is not None, 'Invalid commit SHA')
    return value


def api(endpoint):
    request = urllib.request.Request('https://api.github.com/' + endpoint, headers={
        'Authorization': 'Bearer ' + os.environ['GITHUB_TOKEN'],
        'Accept': 'application/vnd.github+json', 'User-Agent': 'az-tokenizer-pr-hygiene'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def check_paths(paths, allowed):
    require(set(paths).issubset(allowed), 'Changed paths exceed the exact registered scope: '
            + ', '.join(sorted(set(paths) - set(allowed))))
    for name in paths:
        parts = name.lower().split('/')
        require(not any(p in {'.claude', '_repo_setup', '.creds', '.venv', 'artifacts'} for p in parts),
                'Private/runtime folder in diff: ' + name)
        require(not re.search(r'(?i)(\.pat_|\.device_codes|(?:^|/)\.env(?:\.|$)|\.(?:rar|tgz|pt|safetensors)$)', name),
                'Credential/large artifact path in diff: ' + name)


def check_event(pr, card):
    require(pr['title'] == card['title'], 'Title differs from plan')
    require(pr['head']['ref'] == card['branch'], 'Branch differs from plan')
    require(pr['base']['ref'] == 'main', 'Base must be main')
    require(pr['base']['repo']['full_name'].lower() == REPOSITORY, 'Wrong repository')
    require(pr['user']['login'].lower() == card['login'].lower(), 'PR author differs from assigned member')
    marker = f'<!-- git-workflow plan={PLAN_ID} id={card["id"]} -->'
    require(marker in (pr.get('body') or ''), 'Missing current plan marker')
    require(re.search(r'^AMENDMENT: .+', pr.get('body') or '', re.MULTILINE), 'Missing amendment statement')


def verify_dependencies(card, registry, base):
    for dep in card['depends_on']:
        dependency = registry[dep]
        query = urllib.parse.urlencode({'state': 'closed', 'head': 'shahinsafarli:' + dependency['branch'], 'per_page': 100})
        matches = api(f'repos/{REPOSITORY}/pulls?' + query)
        require(len(matches) == 1, 'Dependency PR mapping missing/ambiguous: ' + dep)
        pull = matches[0]
        require(pull['title'] == dependency['title'] and pull.get('merged_at'), 'Dependency not merged: ' + dep)
        marker = f'<!-- git-workflow plan={PLAN_ID} id={dep} -->'
        require(marker in (pull.get('body') or ''), 'Dependency belongs to another plan')
        merged = checked_sha(pull['merge_commit_sha'])
        result = subprocess.run(['git', 'merge-base', '--is-ancestor', merged, base], check=False)
        require(result.returncode == 0, 'Dependency merge is not on this base: ' + dep)


def load_registry(base=None, pid=None):
    path = '.github/pr-plan.yml'
    if base:
        existing = subprocess.run(['git', 'show', base + ':' + path], capture_output=True, check=False)
        raw = existing.stdout if existing.returncode == 0 else None
    else:
        raw = Path(path).read_bytes() if Path(path).exists() else None
    if raw is None:
        require(pid in EARLY or pid is None, 'Full registry must be installed by PR04')
        if pid == 'PR04':
            require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == INITIAL_REGISTRY_SHA256,
                    'PR04 must install the exact supplied registry')
        return EARLY
    plan = json.loads(raw)
    require(plan['meta']['plan_id'] == PLAN_ID, 'Wrong plan version')
    return {p['id']: dict(p, login=plan['members'][p['owner']]['gh']) for p in plan['prs']}


def check_pull(pr):
    match = re.match(r'^(PR\d{2}) ', pr['title'])
    require(match is not None, 'Missing planned PR ID')
    pid = match.group(1)
    base, head = checked_sha(pr['base']['sha']), checked_sha(pr['head']['sha'])
    require(git('rev-parse', 'HEAD').decode().strip() == head, 'Checkout is not the PR head')
    registry = load_registry(base, pid)
    require(pid in registry, 'Unknown PR ID')
    card = registry[pid]
    check_event(pr, card)
    paths = [p.decode('utf-8') for p in git('diff', '--name-only', '-z', base + '...' + head).split(b'\0') if p]
    require(paths, 'Empty contribution')
    check_paths(paths, card['paths'])
    for name in paths:
        path = Path(name)
        require(not path.is_symlink(), 'Symlink in contribution')
        require(not path.is_file() or path.stat().st_size <= 5_000_000, 'File exceeds 5 MB: ' + name)
    commits = git('rev-list', base + '..' + head).decode().splitlines()
    require(len(commits) == 1, 'Use one final author commit per PR')
    message = git('show', '-s', '--format=%B', head).decode()
    require(message.splitlines()[0] == card['commits'][0], 'Commit subject differs from plan')
    require('co-authored-by:' not in message.lower(), 'Co-author trailers are not allowed')
    verify_dependencies(card, registry, base)
    print(pid + ': exact title, branch, author, paths, commit and merged dependencies verified')
    return int(pid[2:]) >= 9


def self_test():
    import copy
    import unittest

    class Guards(unittest.TestCase):
        def test_exact_paths(self):
            check_paths(['LICENSE'], EARLY['PR01']['paths'])
            with self.assertRaises(ValueError): check_paths(['LICENSE.extra'], EARLY['PR01']['paths'])
            with self.assertRaises(ValueError): check_paths(['.pat_FA.txt'], ['.pat_FA.txt'])

        def test_metadata(self):
            card = EARLY['PR01']
            good = {'title': card['title'], 'head': {'ref': card['branch']},
                    'base': {'ref': 'main', 'repo': {'full_name': REPOSITORY}},
                    'user': {'login': card['login']},
                    'body': f'<!-- git-workflow plan={PLAN_ID} id=PR01 -->\nAMENDMENT: none'}
            check_event(good, card)
            for key, value in [('title', 'PR01 wrong'), ('user', {'login': 'shahinsafarli'}), ('body', 'legacy')]:
                bad = copy.deepcopy(good)
                bad[key] = value
                with self.assertRaises(ValueError): check_event(bad, card)

        def test_revision_and_initial_scope(self):
            with self.assertRaises(ValueError): checked_sha('--help')
            self.assertEqual(set(EARLY), {'PR01', 'PR02', 'PR03', 'PR04'})
            self.assertEqual(len(EARLY['PR01']['paths']), 5)
            self.assertEqual(EARLY['PR01']['title'], 'PR01 \u00b7 ci(repo): ci foundation')
            self.assertEqual(re.match(r'^(PR\d{2}) ', EARLY['PR01']['title']).group(1), 'PR01')

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
    require(result.wasSuccessful(), 'Hygiene regression tests failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text(encoding='utf-8'))
    if 'pull_request' in event:
        source_ready = check_pull(event['pull_request'])
    else:
        load_registry()
        source_ready = Path('src/utils.py').exists() and Path('tests/test_core.py').exists()
        print('Main push: infrastructure checks; source suite readiness=' + str(source_ready))
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
            output.write('source_ready=' + str(source_ready).lower() + '\n')


if __name__ == '__main__':
    main()
