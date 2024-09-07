import argparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
from tqdm import tqdm

from selenium.webdriver.chrome.service import Service

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

max_width = max_height = 1024

from mboxreader import extract_emails, email_to_combined_content

chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--disable-extensions");
chrome_options.add_argument("--dns-prefetch-disable");

# Enable verbose logging for ChromeDriver
service = Service(executable_path='/snap/bin/chromium.chromedriver')
service.command_line_args()  # This is where args for ChromeDriver can be modified
# Creating WebDriver with logging
driver = webdriver.Chrome(service=service, options=chrome_options)

def render_email_to_image(email_content, output_path, tmp_path):

    # Use file otherwise doesn't render properly
    with open(tmp_path, 'w') as f:
        f.write(email_content)

    try:
        driver.get("file://" + tmp_path)
    except Exception as e:
        print(e)
        print('skipping email')
        return
        
    # Determine the content's width and height
    width = driver.execute_script("return document.body.scrollWidth") + 0.2*driver.execute_script("return document.body.offsetWidth")
    height = driver.execute_script("return document.body.scrollHeight") + 0.2*driver.execute_script("return document.body.offsetHeight")

    #print('width=',width,' / height=',height)

    width = min(max_width, width)
    height = min(max_height, height)
    
    # Resize the window to fit the content
    driver.set_window_size(width, height)
    
    #driver.get("data:text/html;charset=utf-8," + email_content)

    driver.save_screenshot(output_path)

def process_mbox(mbox_path, output_dir, content_types, nemails, start_from):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    emails = extract_emails(mbox_path)

    for i, email in tqdm(enumerate(emails)):
        if i < start_from:
            continue
        combined_content = email_to_combined_content(email, content_types)
        output_path = os.path.join(output_dir, f'email_{i}.png')
        render_email_to_image(combined_content, output_path, tmp_path='test.html')

        if i == nemails:
            break

def main():
    parser = argparse.ArgumentParser(description='Render emails from an MBOX file to images.')
    parser.add_argument('mbox', help='Path to the input MBOX file')
    parser.add_argument('output_dir', help='Directory to save the output images')
    parser.add_argument(
        '--content-types',
        nargs='+',
        choices=['plain', 'html', 'images', 'richtext', 'pdf', 'calendar'],
        default=['plain', 'html', 'images', 'richtext', 'pdf', 'calendar'],
        help='Specify which content types to render (default: all)'
    )
    parser.add_argument('--nemails', type=int, default=-1 ,help='max number of emails to extract')
    parser.add_argument('--start-from', type=int, default=0, help='email number to start from')

    args = parser.parse_args()

    process_mbox(args.mbox, args.output_dir, args.content_types, args.nemails, args.start_from)

    driver.quit()
    
if __name__ == "__main__":
    main()
