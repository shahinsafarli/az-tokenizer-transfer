# Start here

You have been asked to run an experiment on Google Colab. **You do not need
to understand the research, and you do not need to know Python.** You need a
Google account and a web browser.

---

## Which notebook do I open?

**Open `colab_t4_pilot.ipynb`.** That is the one. Ignore the rest.

| File | Who it is for |
|---|---|
| **`colab_t4_pilot.ipynb`** | **← YOU. Free Colab, ~5–6 h, costs nothing.** |
| `colab_t4.ipynb` | A longer free-Colab run (~8 h+), for later |
| `a100.ipynb` | The full experiment on the university's A100 |
| `vastai_h100.ipynb` | The full experiment on a rented GPU (~$14) |

---

## Getting it open — six clicks

1. Go to **https://colab.research.google.com**
2. Sign in with your Google account.
3. **File → Upload notebook**
4. Drag in **`colab_t4_pilot.ipynb`** (from the folder you were sent).
5. **Runtime → Change runtime type** → *Hardware accelerator*: **T4 GPU** →
   **Save**. Leave "High-RAM" **off**.
6. Start reading from the top of the notebook and follow it. It explains
   every step as you go.

The notebook will ask you to upload the project **`.zip`** file — that is the
other thing you were sent. Keep it handy.

---

## How Colab works, in 60 seconds

- A notebook is a list of **cells**. Grey cells are code; white cells are
  explanation.
- To run a code cell: click it, press **Shift + Enter**.
- A spinning circle on the left means it is working. A number like `[7]`
  means it finished.
- **Run the cells in order, top to bottom.** Do not skip ahead.
- Colab gives you a free GPU in the cloud. It is not using your computer, so
  your laptop can be doing anything else — but **keep the browser tab open**,
  or Colab will disconnect you.

---

## The two slow steps

| Step | How long | What to do |
|---|---|---|
| **Step 7** — build the tokenizer | **1–2 hours** | Start it, leave the tab open, go do something else |
| **Step 9** — the 12 training runs | **3–4 hours** | Same. It saves as it goes |

**Total: about 5–6 hours.** You do **not** have to do it in one sitting. See
below.

---

## If you get disconnected (this is normal and it is fine)

Colab drops free sessions sometimes. **You will not lose your work** —
Step 3 saves everything to your Google Drive.

To carry on:

1. Reopen the notebook.
2. Re-run **Step 1, 2, 3, 4** (they are quick).
3. Jump straight to the step you were on and run it again.

Finished work is detected and skipped automatically. If 7 of the 12 training
runs were done, it does the remaining 5.

**You can also stop on purpose** — finish Step 8, close the tab, and come
back tomorrow for Step 9.

---

## What to send back

At the very end, **Step 12** creates a file called **`pilot_results.zip`** and
downloads it to your computer. **Send that file back.** That is everything.

---

## When to stop and ask

Some cells print a message in capitals saying **STOP**. If you see one:
**stop, copy the output, and send it.** Do not try to fix it and do not
continue past it.

Those messages are not errors in the notebook — they are the experiment
telling you it has found something that has to be looked at by a human before
it is worth spending more hours. Hitting one is a useful outcome, not a
failure.

---

## Common problems

| What you see | What to do |
|---|---|
| **"NO GPU"** | Runtime → Change runtime type → **T4 GPU** → Save. Re-run the cell. |
| **"Runtime disconnected"** | Reconnect, re-run Steps 1–4, continue where you were. |
| **"CUDA out of memory"** | Runtime → **Restart session**. Re-run Steps 2–5, continue. Do not change any settings. |
| **`ModuleNotFoundError`** | Re-run Step 4, then Step 2. |
| **Step 7 looks frozen** | It is doing quiet CPU work. Give it a full 2 hours before worrying. |
| **Step 9 stopped early** | Normal — it has a built-in time limit. Just run that cell again. |
| **A cell asks about Google Drive** | Click through and **Allow**. This is Step 3 saving your work. |

---

## One thing to please not do

Do not edit any file in `configs/`. The experiment checks those files against
a recorded fingerprint before it runs anything, and it will refuse to start if
they have changed. That check is there on purpose — it is what makes the
results trustworthy. If a cell tells you a config check failed, send that
message back rather than editing anything.

---

Thank you — this genuinely saves a week of GPU time.
