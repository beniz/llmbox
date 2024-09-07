import sys
import logging
import mailbox
import math

import html2text
h = html2text.HTML2Text()
html_clues = ['<html', '<HTML', '<body', '<BODY', '<div', '<DIV']
keep_types = ['text/plain', 'text/html']

##TODO: argparse

def get_maildata(msg):
    message = mailbox.Message(msg)
    try:
        keys = message.keys()
        if 'X-Spam-Status' in keys:
            del message['X-Spam-Status']
            del message['X-Spam-Level']
            del message['X-Spam-Checker-Version']
            
        body = message.as_string()
        for hc in html_clues:
            if hc in body:
                body = h.handle(body)
                break

    except Exception as e:
        print(e)
        return None
    datarow = {"input": body}
    #print(datarow)
    return datarow


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

def get_prediction(input_text: str):
    url = "http://localhost:8235/predict"
    
    # Prepare the data as a dictionary
    data = {"input": input_text}
    
    # Make the POST request
    s = requests.Session()
    retries = Retry(total=400,
                backoff_factor=0.1,
                status_forcelist=[ 500, 502, 503, 504 ])
    response = s.post(url, json=data, headers={"Content-Type": "application/json"}, timeout=600)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        prediction = response.json().get("prediction")
        return prediction
    else:
        # Handle the error
        raise Exception(f"Request failed with status code {response.status_code}")

if __name__ == "__main__":

    # logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='/tmp/.llmbox.log', level=logging.INFO)
    
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
    
    preproc_msg = get_maildata(msg)

    count_ham = 0
    count_spam = 0
    n = 8000
    nchunks_limit = 4
    preproc_input = preproc_msg['input']
    logger.info('processing email of size %s', len(preproc_input))

    if len(preproc_input) > n:
        chunks = [preproc_input[i:i+n] for i in range(0, len(preproc_input), n)]
        nchunks = len(chunks)
        #print('nchunks=', nchunks)
        logger.info('number of chunks %s', nchunks)
        if nchunks > nchunks_limit:
            nchunks = nchunks_limit
            logger.info('number of chunks limited to %s', nchunks)
        for i in range(nchunks):
            chunk_msg = chunks[i]
            
            logger.info('chunk size %s', len(chunk_msg))
            
            chunk_msg = {'input': chunk_msg}

            fmt_msg= formatting_prompts_func(chunk_msg)

            prediction = get_prediction(fmt_msg)
            

            if prediction == 'Spam':
                logger.info('got Spam status')
                count_spam += 1
            elif prediction == 'Ham':
                count_ham +=1
                logger.info('got Ham status')
            else:
                logger.info('got prediction %s', prediction)

            # if count_ham > half chunks, skip remaining chunks
            if count_spam >= math.ceil(nchunks+1/2) or count_ham >= math.ceil(nchunks+1/2):
                logger.info('skipping remaining %s chunks', nchunks-count_ham-count_spam)
                break
                
    else:
        fmt_msg= formatting_prompts_func(preproc_msg)
        
        prediction = get_prediction(fmt_msg)

        if prediction == 'Spam':
            logger.info('got Spam status')
            count_spam += 1
        elif prediction == 'Ham':
            count_ham +=1
            logger.info('got Ham status')
        else:
            logger.info('got prediction %s', prediction)
        
    spam_header = 'X-Spam-Status'
    if count_ham >= count_spam:
        tag = 'No'
    else:
        tag = 'Yes'
    logger.info('final status: %s', tag)

    # add header correctly to message
    message = mailbox.Message(msg)
    del message[spam_header]
    message[spam_header] = tag
    sys.stdout.write(message.as_string(unixfrom=True))
