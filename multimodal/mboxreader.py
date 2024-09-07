import mailbox
from email.utils import make_msgid, getaddresses
from email.header import decode_header
import chardet
from tqdm import tqdm
from icalendar import Calendar

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def decode_mime_words(s):
    """Decodes MIME-encoded words to a readable string."""
    decoded_parts = []
    for part, encoding in decode_header(s):
        if isinstance(part, bytes):
            try:
                # Attempt to decode using the specified encoding
                decoded_parts.append(part.decode(encoding or 'utf-8', errors='replace'))
            except (LookupError, UnicodeDecodeError):
                # Fallback in case of an unknown encoding or decoding error
                #print('failed extracting field 
                decoded_parts.append(part.decode('utf-8', errors='replace'))
        else:
            decoded_parts.append(part)
    return ''.join(decoded_parts)

def extract_emails(mbox_path):
    logger.debug('Extracting emails')
    mbox = mailbox.mbox(mbox_path)
    emails = []
    for message in tqdm(mbox):
        emails.append(message)
        #break # debug: single email
    return emails

def detect_charset(message):
    logger.debug('Detecting charset')
    charsets = message.get_charsets()
    if charsets:
        charset = charsets[0]
    else:
        text = message.get_payload(decode=False)
        if text:
            try:
                charset = charset.detect(text)['encoding']
            except:
                logger.debug('Failed charset.detect')
                charset = 'utf-8'
        else:
            charset = 'utf-8' # default
            logger.debug('Defaulting to utf-8 charset')
    ## hack: ascii -> latin-1
    if charset and charset.lower() == 'ascii':
        logger.debug('Hack: turning ascii charset into latin-1')
        charset = 'latin-1'
    logger.debug('Charset detected: %s', charset)
    return charset

def get_plain_payload(message):
    logger.debug('getting plain text payload')
    encoding = detect_charset(message)
    #print('plain encoding=', encoding)
    try:
        plain_text = message.get_payload(decode=True).decode(encoding)
    except Exception as e:
        if "codec can't decode byte 0xe" in str(e):
           plain_text = message.get_payload(decode=True).decode('latin-1')
        else:
            raise e
    
    #print('plain_text=',plain_text)
    return plain_text

def get_html_payload(message):
    logger.debug('getting html payload')
    encoding = detect_charset(message)
    #print('html encoding=', encoding)
    try:
        html_text = message.get_payload(decode=True).decode(encoding)
    except Exception as e:
        if "codec can't decode byte 0xe" in str(e):
           html_text = message.get_payload(decode=True).decode('latin-1')
        else:
            raise e
    return html_text

def parse_calendar_data(calendar_data):
    """Parse and render calendar data into HTML."""
    cal = Calendar.from_ical(calendar_data)
    events_html = "<div class='calendar-events'>"
    
    for component in cal.walk():
        if component.name == "VEVENT":
            summary = component.get('summary')
            start = component.get('dtstart').dt
            end = component.get('dtend').dt
            location = component.get('location')
            description = component.get('description')
            
            events_html += f"""
            <div class='calendar-event'>
                <p><strong>Event:</strong> {summary}</p>
                <p><strong>Start:</strong> {start}</p>
                <p><strong>End:</strong> {end}</p>
                <p><strong>Location:</strong> {location}</p>
                <p><strong>Description:</strong> {description}</p>
            </div>
            <hr/>
            """
    
    events_html += "</div>"
    return events_html

def email_to_combined_content(message, content_types):

    # Extract headers
    from_header = decode_mime_words(message.get('From', ''))
    to_header = decode_mime_words(message.get('To', ''))
    subject_header = decode_mime_words(message.get('Subject', ''))
    #date_header = decode_mime_words(message.get('Date', ''))
    cc_header = decode_mime_words(message.get('Cc', ''))

    # Start building the HTML content
    ##broken: style="border-bottom: 1px solid #ccc; padding-bottom: 10px; margin-bottom: 10px;">
    header_html = f"""
    <div> 
        <p><strong>From:</strong> {from_header}</p>
        <p><strong>To:</strong> {to_header}</p>
        <p><strong>Subject:</strong> {subject_header}</p>
        <p><strong>Cc:</strong> {cc_header}</p>
    </div>
    """

    #print('header_html=', header_html)
    
    plain_text = ""
    html_text = ""
    images = []
    pdf = ""
    rich_text = ""
    calendar_text = ""

    if message.is_multipart():
        logger.debug('multipart message')
        for part in message.walk():
            content_type = part.get_content_type()

            try:
                if "text/plain" in content_type and 'plain' in content_types:
                    plain_text += get_plain_payload(part)
                    
                elif "text/html" in content_type and 'html' in content_types:
                    #html_text += part.get_payload(decode=True).decode()
                    html_text += get_html_payload(part)
                    
                elif ("image/jpeg" in content_type or "image/jpg" in content_type):
                    #print('got image')
                    image_data = part.get_payload(decode=False).encode()
                    #print('image data=', image_data)
                    image_cid = make_msgid(domain="example.com")
                    image_src = f"data:{content_type};base64,{image_data.encode('base64')}"
                    images.append((image_cid, image_src))

                #elif "text/css" in content_type:
                #    css += f"<style>{part.get_payload(decode=True).decode()}</style>"
                elif "text/calendar" in content_type and 'calendar' in content_types:
                    calendar_part = part.get_payload(decode=True).decode()
                    calendar_text += parse_calendar_data(calendar_part)
                
                #elif "application/pdf" in content_type and 'pdf' in content_types:
                #    pdf = part.get_payload(decode=True).decode()
            
                elif ("text/richtext" in content_type or "text/enriched" in content_type) and 'richtext' in content_types:
                    rich_text += part.get_payload(decode=True).decode()
            except Exception as e:
                #logger.error('failed getting payload', e)
                continue

    else:
        logger.debug('single part message')
        try:
            content_type = message.get_content_type()
            if "text/plain" in content_type and 'plain' in content_types:
                plain_text = get_plain_payload(message)
            elif "text/html" in content_type and 'html' in content_types:
                html_text = get_html_payload(message)
        except Exception as e:
            print('failed getting plain or html payload', e)

    # Combine headers and body into one HTML structure
    combined_content = f"<html><body>{header_html}"
            
    # Combine content types into one HTML structure
    if 'html' in content_types and len(html_text):
         combined_content += html_text
    elif 'plain' in content_types and len(plain_text):
        combined_content += f"<pre>{plain_text}</pre>"
    
    # Embed inline images
    if 'images' in content_types:
        for cid, src in images:
            combined_content += combined_content.replace(f'cid:{cid}', src)
    
    # Add PDF embedding
    if 'pdf' in content_types and pdf:
        combined_content += f'<object data="data:application/pdf;base64,{pdf.encode("base64").decode("utf-8")}" type="application/pdf" width="100%" height="800px"></object>'
    
    # Add rich text content
    if 'richtext' in content_types and rich_text:
        combined_content += f"<pre>{rich_text}</pre>"

    # Add calendar
    if 'calendar' in content_types and calendar_text:
        combined_content += calendar_text
        
    combined_content += "</body></html>"
    
    return combined_content

