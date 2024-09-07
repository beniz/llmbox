import sys
import logging
import mailbox
import math

prompt = """Below is an email message as input. Decide\
  whether it is spam or ham. In response, if spam write as Spam otherwise write Ham.

### Input: {}

### Response: """

def formatting_prompts_func(msg):
    inputs       = msg["input"]
    text = prompt.format(inputs)
    return text

import requests
from requests.adapters import Retry

def get_prediction(img_path: str):
    url = "http://localhost:8236/predict"
    
    # Prepare the input data
    with open(img_path, "rb") as image_file:
        files = {"file": image_file}
    

        # Make the POST request
        s = requests.Session()
        retries = Retry(total=400,
                        backoff_factor=0.1,
                        status_forcelist=[ 500, 502, 503, 504 ])
        response = s.post(url, files=files, timeout=600)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        prediction = response.json().get("prediction")
        confidence = response.json().get("confidence")
        return prediction, confidence
    else:
        # Handle the error
        raise Exception(f"Request failed with status code {response.status_code}")

if __name__ == "__main__":

    # logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='/tmp/.llmbox_multimodal.log', level=logging.INFO)
    
    # use stdin if it's full
    if not sys.stdin.isatty():
        input_stream = sys.stdin

    # otherwise, read the given filename
    else:
        try:
            input_filename = sys.argv[1]
        except IndexError:
            message = 'need filename as first argument if stdin is not full'
            raise IndexError(message)
        else:
            input_stream = open(input_filename, 'rU')

    msg = ''
    for line in input_stream:
        msg += line

    # turn email into image
    from mbox2images import email_to_combined_content, render_email_to_image
    output_filename = '/tmp/tmp_email.png'
    content_types = ['plain', 'html', 'images', 'richtext', 'pdf', 'calendar']
    message = mailbox.Message(msg)
    combined_content = email_to_combined_content(message, content_types)
    render_email_to_image(combined_content, output_filename, tmp_path='/tmp/tmp_email.html')
    
    count_spam = 0
    count_ham = 0 
    prediction, confidence = get_prediction(output_filename)

    if prediction == 'spam':
        logger.info('got spam status')
        count_spam += 1
    elif prediction == 'ham':
        count_ham +=1
        logger.info('got ham status')
    else:
        logger.info('got prediction %s', prediction)
        
    spam_header = 'X-Spam-Status'
    if count_ham >= count_spam:
        tag = 'No'
    else:
        tag = 'Yes'
    spam_score_header = 'X-Spam-Score'
    logger.info('final status: %s', tag)
    #print(tag)
    #print(confidence)

    # add header correctly to message
    message = mailbox.Message(msg)
    del message[spam_header]
    message[spam_header] = tag
    message[spam_score_header] = str(confidence)
    sys.stdout.write(message.as_string(unixfrom=True))
