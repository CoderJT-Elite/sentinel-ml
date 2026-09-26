"""Record screen captures of the live app with Playwright (deck clips, video scenes).

    python scripts/record_demo.py http://localhost:8000 out_dir [gate|run|explain|deploy|drift ...]

Drives the real UI with a visible cursor. Needs the full stack (docker compose up) for the deploy clip, which uses
the Docker socket. The "run" clip trains a real model (about two and a half minutes); the later clips open that run.
Output is one .webm per clip in out_dir; convert with ffmpeg (see the README of the submission kit).
"""
import json
import pathlib
import sys

from playwright.sync_api import sync_playwright

url, out = sys.argv[1], pathlib.Path(sys.argv[2])
wanted = sys.argv[3:] or ["gate", "run", "explain", "deploy", "drift"]
out.mkdir(parents=True, exist_ok=True)
VIEW = {"width": 1500, "height": 900}

CURSOR = """
(() => {
  const d = document.createElement('div');
  d.style.cssText = 'position:fixed;z-index:99999;left:-50px;top:-50px;width:24px;height:24px;border-radius:50%;' +
    'border:2px solid #d8001b;background:rgba(216,0,27,.15);pointer-events:none;transform:translate(-50%,-50%)';
  document.addEventListener('DOMContentLoaded', () => document.body.appendChild(d));
  addEventListener('mousemove', e => { d.style.left = e.clientX + 'px'; d.style.top = e.clientY + 'px'; }, true);
  addEventListener('mousedown', () => { d.style.background = 'rgba(216,0,27,.65)'; }, true);
  addEventListener('mouseup', () => { d.style.background = 'rgba(216,0,27,.15)'; }, true);
})();
"""


def glide(pg, selector, pause=250):
    """Move the pointer to the element in a visible arc, then click it."""
    loc = pg.locator(selector).first
    loc.scroll_into_view_if_needed()
    pg.wait_for_timeout(200)
    box = loc.bounding_box()
    pg.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=28)
    pg.wait_for_timeout(pause)
    pg.mouse.down()
    pg.mouse.up()
    pg.wait_for_timeout(pause)


def wait_idle(pg, timeout=900_000):
    pg.wait_for_function("window.__S && window.__S.status !== 'running'", timeout=timeout)


def session(browser, name):
    ctx = browser.new_context(viewport=VIEW, record_video_dir=str(out / "_raw"), record_video_size=VIEW)
    ctx.add_init_script(CURSOR)
    return ctx, ctx.new_page()


def finish(ctx, pg, name):
    path = pg.video.path()
    ctx.close()
    dest = out / f"{name}.webm"
    pathlib.Path(path).replace(dest)
    print("recorded", dest)


def open_run(pg, run_id):
    glide(pg, ".tab[data-id=data]")
    pg.wait_for_selector(f".open[data-id='{run_id}']")
    glide(pg, f".open[data-id='{run_id}']")
    pg.wait_for_selector(".tab[aria-selected=true]")
    pg.wait_for_timeout(900)


with sync_playwright() as p:
    b = p.chromium.launch()
    state = out / "run.json"

    if "gate" in wanted:
        ctx, pg = session(b, "gate")
        pg.goto(url)
        pg.wait_for_selector(".dsc")
        pg.wait_for_timeout(1500)
        glide(pg, ".dsc:has-text('broken sensor log')")
        pg.wait_for_timeout(800)
        glide(pg, "#run")
        pg.wait_for_timeout(1500)
        wait_idle(pg, 90_000)
        pg.wait_for_timeout(3500)
        finish(ctx, pg, "gate")

    if "run" in wanted:
        ctx, pg = session(b, "run")
        pg.goto(url)
        pg.wait_for_selector(".dsc")
        pg.wait_for_timeout(1500)
        glide(pg, ".dsc:has-text('C-MAPSS')")
        pg.wait_for_timeout(600)
        glide(pg, "#run")
        wait_idle(pg)
        pg.wait_for_timeout(2500)
        run_id = pg.evaluate("window.__S.result.run_id")
        state.write_text(json.dumps({"run_id": run_id}))
        finish(ctx, pg, "run")

    run_id = json.loads(state.read_text())["run_id"] if state.exists() else None

    if "explain" in wanted and run_id:
        ctx, pg = session(b, "explain")
        pg.goto(url)
        pg.wait_for_selector(".dsc")
        pg.wait_for_timeout(1000)
        open_run(pg, run_id)
        glide(pg, ".tab[data-id=board]")
        pg.wait_for_timeout(2200)
        glide(pg, ".tab[data-id=explain]")
        pg.wait_for_timeout(2200)
        glide(pg, ".tab[data-id=fleet]")
        pg.wait_for_timeout(1500)
        glide(pg, "tr.clickable[data-id='76']")
        pg.wait_for_timeout(2500)
        pg.mouse.wheel(0, 380)
        pg.wait_for_timeout(1200)
        box = pg.locator(".sim-slider").first.bounding_box()
        pg.mouse.move(box["x"] + box["width"] * 0.8, box["y"] + box["height"] / 2, steps=25)
        for v in [1.5, 1.0, 0.5, 0.0, -0.5]:
            pg.locator(".sim-slider").first.evaluate("(e, v) => { e.value = v; e.dispatchEvent(new Event('input', {bubbles: true})); }", v)
            pg.wait_for_timeout(700)
        pg.wait_for_timeout(1500)
        finish(ctx, pg, "explain")

    if "deploy" in wanted and run_id:
        ctx, pg = session(b, "deploy")
        pg.goto(url)
        pg.wait_for_selector(".dsc")
        pg.wait_for_timeout(1000)
        open_run(pg, run_id)
        glide(pg, ".tab[data-id=deploy]")
        pg.wait_for_timeout(2500)
        if pg.query_selector("#dk"):
            glide(pg, "#dk")
            pg.wait_for_function("document.querySelector('.view .note') !== null", timeout=300_000)
        pg.wait_for_timeout(3000)
        finish(ctx, pg, "deploy")

    if "drift" in wanted and run_id:
        ctx, pg = session(b, "drift")
        pg.goto(url)
        pg.wait_for_selector(".dsc")
        pg.wait_for_timeout(1000)
        open_run(pg, run_id)
        glide(pg, ".tab[data-id=monitor]")
        pg.wait_for_selector(".pill.STABLE, .pill.WATCH, .pill.RETRAIN", timeout=120_000)
        pg.wait_for_timeout(2000)
        box = pg.locator("#sig").bounding_box()
        pg.mouse.move(box["x"] + 6, box["y"] + box["height"] / 2, steps=20)
        for i in range(1, 6):
            pg.wait_for_selector("#sig", timeout=60_000)
            pg.evaluate(f"(() => {{ const s = document.querySelector('#sig'); s.value = {i}; s.dispatchEvent(new Event('input')); s.dispatchEvent(new Event('change')); }})()")
            pg.mouse.move(box["x"] + 6 + i * box["width"] / 7, box["y"] + box["height"] / 2, steps=6)
            pg.wait_for_timeout(900)
        pg.wait_for_selector(".pill.RETRAIN", timeout=120_000)
        pg.wait_for_timeout(2500)
        glide(pg, "#rt")
        pg.wait_for_selector(".view table:has-text('Challenger')", timeout=300_000)
        pg.wait_for_timeout(1500)
        pg.mouse.wheel(0, 320)
        pg.wait_for_timeout(3500)
        finish(ctx, pg, "drift")
    b.close()
