"""Capture screenshots of the running app with Playwright (submission snapshots, deck, video).

    python scripts/snap.py http://localhost:8000 run-abc123 out_dir

Drives the real UI: the quality gate refusing a broken sample, then every tab of a finished run,
the container smoke test, and the drift -> retrain scenario.
"""
import sys

from playwright.sync_api import sync_playwright

url, run_id, out = sys.argv[1], sys.argv[2], sys.argv[3]
VIEW = {"width": 1500, "height": 900}


def wait_idle(pg, timeout=900_000):
    pg.wait_for_function("window.__S && window.__S.status !== 'running'", timeout=timeout)


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport=VIEW, device_scale_factor=1.5)
    pg.goto(url)
    pg.wait_for_selector(".dsc")
    pg.wait_for_timeout(1200)
    pg.screenshot(path=f"{out}/data.png")

    # 1) the quality gate refusing bad data
    pg.click(".dsc:has-text('broken sensor log')")
    pg.click("#run")
    pg.wait_for_timeout(1500)
    wait_idle(pg, 60_000)
    pg.wait_for_timeout(1200)
    pg.screenshot(path=f"{out}/gate_fail.png")
    print("shot gate_fail")

    # 2) a finished run, tab by tab
    pg.click(".tab[data-id=data]")
    pg.wait_for_selector(f".open[data-id='{run_id}']")
    pg.click(f".open[data-id='{run_id}']")
    pg.wait_for_selector(".tab[aria-selected=true]")
    pg.wait_for_timeout(1200)
    for t in ["quality", "board", "explain", "fleet"]:
        pg.click(f".tab[data-id={t}]")
        pg.evaluate("window.scrollTo(0, 0)")
        pg.wait_for_timeout(1000)
        pg.screenshot(path=f"{out}/{t}.png")
        print("shot", t)

    # what-if: a borderline engine, one driver moved back to its baseline, re-scored by the champion's own trees
    pg.click(".tab[data-id=fleet]")
    pg.click("tr.clickable[data-id='91']")
    pg.wait_for_timeout(600)
    pg.evaluate("document.querySelector('.sim-slider').value = 0; document.querySelector('.sim-slider').dispatchEvent(new Event('input', {bubbles: true}))")
    pg.evaluate("document.querySelector('#simCheck').scrollIntoView({block: 'end'})")
    pg.wait_for_timeout(600)
    pg.screenshot(path=f"{out}/whatif.png")
    print("shot whatif", pg.evaluate("[...document.querySelectorAll('.whatif, .sim-rows, .sim-out, #simCheck')].map(e => [e.className || e.id, ...Object.values(e.getBoundingClientRect().toJSON()).slice(0, 4).map(Math.round)])"))

    pg.click(".tab[data-id=deploy]")
    pg.evaluate("window.scrollTo(0, 0)")
    pg.wait_for_timeout(600)
    pg.screenshot(path=f"{out}/deploy_before.png")
    if pg.query_selector("#dk"):
        pg.click("#dk")
        pg.wait_for_function("document.querySelector('.view .note') !== null", timeout=300_000)
        pg.wait_for_timeout(800)
    pg.screenshot(path=f"{out}/deploy.png")
    print("shot deploy")

    pg.set_viewport_size({"width": 1500, "height": 1250})   # tall enough for the whole Monitor tab, no full-page stitching
    pg.evaluate("window.scrollTo(0, 0)")
    pg.click(".tab[data-id=monitor]")
    pg.wait_for_selector(".pill.STABLE, .pill.WATCH, .pill.RETRAIN", timeout=120_000)
    pg.wait_for_timeout(800)
    pg.screenshot(path=f"{out}/monitor_0.png")
    idx = pg.evaluate("[0,0.25,0.5,1,1.5,2,3].indexOf(2)")
    pg.evaluate(f"(() => {{ const s = document.querySelector('#sig'); s.value = {idx}; s.dispatchEvent(new Event('input')); s.dispatchEvent(new Event('change')); }})()")
    pg.wait_for_selector(".pill.RETRAIN", timeout=120_000)
    pg.wait_for_timeout(800)
    pg.screenshot(path=f"{out}/monitor_2.png")
    pg.click("#rt")
    pg.wait_for_selector(".view table:has-text('Challenger')", timeout=300_000)
    pg.wait_for_timeout(800)
    pg.evaluate("window.scrollTo(0, 0)")
    pg.screenshot(path=f"{out}/monitor_retrain.png")
    print("shot monitor")
    b.close()
