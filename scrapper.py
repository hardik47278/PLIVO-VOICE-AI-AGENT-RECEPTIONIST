"""
IEI India Website Scraper using Playwright
Scrapes dynamically rendered content from ieindia.org
"""

import asyncio
import json
import re
import time
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright

BASE_URL = "https://www.ieindia.org"

# Deduplicated list — 63 unique URLs (20 duplicates removed from original 83-line list;
# one malformed line containing two concatenated URLs was split into two entries)
URLS_TO_SCRAPE = [
    ("Certification - Chartered Engineer",   "https://www.ieindia.org/web/certification#chartered"),
    ("Education CPD - Academics",            "https://www.ieindia.org/web/education-cpd#academics"),
    ("Education CPD - Eligibility",          "https://www.ieindia.org/web/education-cpd#eligibility"),
    ("Education CPD - Notice",               "https://www.ieindia.org/web/education-cpd#notice"),
    ("Education CPD - Accreditation",        "https://www.ieindia.org/web/education-cpd#accreditation"),
    ("Education CPD - Recognition",          "https://www.ieindia.org/web/education-cpd#recognition"),
    ("Education CPD - Guidelines",           "https://www.ieindia.org/web/education-cpd#guidelines"),
    ("Education CPD - Reg SecB",             "https://www.ieindia.org/web/education-cpd#reg-secb"),
    ("Education CPD - Lab Exp",              "https://www.ieindia.org/web/education-cpd#lab-exp"),
    ("Seminar Workshop",                     "https://www.ieindia.org/web/seminar-workshop"),
    ("Webinar",                              "https://www.ieindia.org/web/webinar"),
    ("Education CPD - SIM",                  "https://www.ieindia.org/web/education-cpd#SIM"),
    ("Education CPD - Fees",                 "https://www.ieindia.org/web/education-cpd#fees"),
    ("Education CPD - Download",             "https://www.ieindia.org/web/education-cpd#download"),
    ("Research - Overview",                  "https://www.ieindia.org/web/research#overview"),
    ("Research - Key Highlights",            "https://www.ieindia.org/web/research#key-highlights"),
    ("Research - Eligibility Criteria",      "https://www.ieindia.org/web/research#eligibility-criteria"),
    ("Research - Apply",                     "https://www.ieindia.org/web/research#apply"),
    ("Research - Funded",                    "https://www.ieindia.org/web/research#funded"),
    ("Prize Award - IEA",                    "https://www.ieindia.org/web/prize-award#IEA"),
    ("Prize Award - EEEA",                   "https://www.ieindia.org/web/prize-award#EEEA"),
    ("Prize Award - YEA",                     "https://www.ieindia.org/web/prize-award#YEA"),
    ("Prize Award - PBJP",                   "https://www.ieindia.org/web/prize-award#PBJP"),
    ("Prize Award - SAIL",                   "https://www.ieindia.org/web/prize-award#SAIL"),
    ("Prize Award - COAL",                   "https://www.ieindia.org/web/prize-award#COAL"),
    ("Technical Activity - National Convention", "https://www.ieindia.org/web/technical-activity#nationalconvention"),
    ("Technical Activity - Seminar",          "https://www.ieindia.org/web/technical-activity#seminar"),
    ("Technical Activity - Webinar",          "https://www.ieindia.org/web/technical-activity#webinar"),
    ("Technical Activity - Statutory Days",   "https://www.ieindia.org/web/technical-activity#statutorydays"),
    ("TA Guidelines (ipanel)",               "https://ipanel.ieindia.org/webui/IEI-TAGuidelines.html"),
    ("Publication - Annual Reports",          "https://www.ieindia.org/web/publication#annualreports"),
    ("Publication - News",                    "https://www.ieindia.org/web/publication#news"),
    ("Publication - Epitome",                 "https://www.ieindia.org/web/publication#epitome"),
    ("Publication - Journal",                 "https://www.ieindia.org/web/publication#journal"),
    ("Publication - Schedule Rate",           "https://www.ieindia.org/web/publication#schedulerate"),
    ("Certification - ARBT",                  "https://www.ieindia.org/web/certification#arbt"),
    ("Certification - PE",                    "https://www.ieindia.org/web/certification#pengg"),
    ("Certification - Int PE",                "https://www.ieindia.org/web/certification#intpengg"),
    ("Become Member",                         "https://www.ieindia.org/web/becomemember#"),
    ("Membership - Benefit",                  "https://www.ieindia.org/web/membership#benifit"),
    ("Membership - Grades",                   "https://www.ieindia.org/web/membership#grades"),
    ("Membership - Upgrades",                 "https://www.ieindia.org/web/membership#upgrades"),
    ("iMember",                                "https://www.ieindia.org/web/imember#"),
    ("Membership - Fees",                     "https://www.ieindia.org/web/membership#fees"),
    ("Student Chapters - Schapter",           "https://www.ieindia.org/web/network/studentchapters#schapter"),
    ("Student Chapters - Benefit",            "https://www.ieindia.org/web/network/studentchapters#sbenifit"),
    ("Student Chapters - Join",               "https://www.ieindia.org/web/network/studentchapters#joinCh"),
    ("Student Chapters - Scholar Directory",  "https://www.ieindia.org/web/network/studentchapters#scholardirectory"),
    ("Member Search - Nearest Centre",        "https://www.ieindia.org/web/membersearch?option=find-nearest-centre"),
    ("Advert",                                "https://www.ieindia.org/web/advert#"),
    ("FAQ - Know More",                       "https://www.ieindia.org/web/faq#knownmore"),
    ("Tender (ipanel)",                       "https://ipanel.ieindia.org/webui/IEI-Tender.aspx?v20230717.1"),
    ("About - Engg Divisions",                "https://www.ieindia.org/web/about#enggdiv"),
    ("IEI Council",                           "https://www.ieindia.org/web/iei-council#council"),
    ("About - Royal Charter",                 "https://www.ieindia.org/web/about#royalcharter"),
    ("Bye-laws Regulation",                   "https://www.ieindia.org/web/byelawsRegulation"),
    ("About - Code of Ethics",                "https://www.ieindia.org/web/about#codeofethics"),
    ("Network",                               "https://www.ieindia.org/web/network"),
    ("Guest House Info",                      "https://www.ieindia.org/web/guest-house-info"),
    ("Publication - Books",                   "https://www.ieindia.org/web/publication#books"),
    ("Publication - Technical Volume",        "https://www.ieindia.org/web/publication#technicalvolume"),
    ("Certification - Int PE (alt path)",     "https://www.ieindia.org/web/certification/intPengg"),
    ("Prize Award - NDRF",                    "https://www.ieindia.org/web/prize-award#NDRF"),
]


def clean_text(text):
    """Remove excessive whitespace and blank lines."""
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    # collapse 3+ blank lines to 1
    result, prev_blank = [], False
    for l in lines:
        if l == "":
            if not prev_blank:
                result.append(l)
            prev_blank = True
        else:
            result.append(l)
            prev_blank = False
    return "\n".join(result)

def extract_links(page_links, base_url):
    """Filter and normalise links found on the page."""
    seen, out = set(), []
    for href, text in page_links:
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if "ieindia.org" not in parsed.netloc:
            continue
        if full not in seen:
            seen.add(full)
            out.append({"url": full, "text": text.strip()[:120]})
    return out

async def scrape_page(page, name, url):
    """Navigate to a URL, wait for content, extract text and links."""
    print(f"  Scraping: {name} — {url}")
    result = {"name": name, "url": url, "status": "ok",
              "title": "", "text": "", "links": [], "api_calls": []}
    try:
        # Capture XHR/fetch calls the page makes (reveals internal APIs)
        api_calls = []
        def on_request(req):
            if req.resource_type in ("xhr", "fetch"):
                api_calls.append(req.url)
        page.on("request", on_request)

        resp = await page.goto(url, wait_until="networkidle", timeout=30000)
        result["status_code"] = resp.status if resp else None

        # Extra wait for SPA rendering
        await page.wait_for_timeout(2500)

        # Try to dismiss any cookie/modal overlays
        for sel in ["button:has-text('Accept')", "button:has-text('Close')",
                    "button:has-text('OK')", ".modal-close", "[aria-label='Close']"]:
            try:
                btn = page.locator(sel).first
                if await btn.is_visible(timeout=1000):
                    await btn.click()
                    await page.wait_for_timeout(500)
            except Exception:
                pass

        result["title"] = await page.title()

        # Extract visible text from the main content area
        raw_text = await page.evaluate("""() => {
            // Remove nav, footer, script, style noise
            const remove = ['nav','footer','script','style','noscript',
                            '.navbar','.footer','[aria-hidden="true"]'];
            remove.forEach(s => {
                document.querySelectorAll(s).forEach(el => el.remove());
            });
            return document.body ? document.body.innerText : '';
        }""")
        result["text"] = clean_text(raw_text)[:8000]  # cap at 8k chars

        # Extract all links
        raw_links = await page.evaluate("""() => {
            return Array.from(document.querySelectorAll('a[href]'))
                .map(a => [a.getAttribute('href'), a.innerText.trim()]);
        }""")
        result["links"] = extract_links(raw_links, url)
        result["api_calls"] = list(set(
            u for u in api_calls if "ieindia.org" in u
        ))[:30]

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    return result

async def main():
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()

        for name, url in URLS_TO_SCRAPE:
            data = await scrape_page(page, name, url)
            results.append(data)
            # Polite delay
            await asyncio.sleep(1)

        await browser.close()

    # Save full JSON
    with open("iei_scraped.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save readable text report
    with open("iei_report.txt", "w", encoding="utf-8") as f:
        f.write("IEI INDIA — SCRAPED CONTENT REPORT\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            f.write(f"{'='*70}\n")
            f.write(f"PAGE : {r['name']}\n")
            f.write(f"URL  : {r['url']}\n")
            f.write(f"TITLE: {r.get('title','')}\n")
            f.write(f"STATUS: {r['status']} | HTTP {r.get('status_code','?')}\n")
            f.write("-" * 70 + "\n")

            if r.get("text"):
                f.write("CONTENT:\n")
                f.write(r["text"][:4000] + ("\n...[truncated]" if len(r["text"]) > 4000 else "") + "\n\n")

            if r.get("links"):
                f.write(f"LINKS FOUND ({len(r['links'])}):\n")
                for lnk in r["links"][:40]:
                    f.write(f"  [{lnk['text'][:60]}] {lnk['url']}\n")
                if len(r["links"]) > 40:
                    f.write(f"  ... and {len(r['links'])-40} more\n")
                f.write("\n")

            if r.get("api_calls"):
                f.write(f"INTERNAL API/XHR CALLS ({len(r['api_calls'])}):\n")
                for a in r["api_calls"]:
                    f.write(f"  {a}\n")
                f.write("\n")

            if r.get("error"):
                f.write(f"ERROR: {r['error']}\n\n")

    print("\nDone! Files saved:")
    print("  /home/claude/iei_scraped.json  (full structured data)")
    print("  /home/claude/iei_report.txt    (readable text report)")
    print(f"\nPages scraped: {len(results)}")
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"Successful:    {ok}/{len(results)}")

if __name__ == "__main__":
    asyncio.run(main())