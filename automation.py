from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        context = browser.new_context(ignore_https_errors=True)
        page = context.new_page()
        page.goto('https://www.google.com/search?q=OpenAI+Codex+browser+automation')
        page.wait_for_selector('body')
        page.screenshot(path='results.png')
        browser.close()

if __name__ == '__main__':
    run()
