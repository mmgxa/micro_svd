FROM python:3.9

WORKDIR /code

RUN pip install \
	torch==2.0.1+cpu \
	torchvision==0.15.2+cpu \
	--extra-index-url https://download.pytorch.org/whl/cpu \
	&& rm -rf /root/.cache/pip

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN python -c 'from transformers import GPT2Tokenizer, GPT2LMHeadModel; tokenizer=GPT2Tokenizer.from_pretrained("gpt2"); GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)'

COPY ./model_server.py /code/

CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "80"]