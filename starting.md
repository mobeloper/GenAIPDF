
Create the virtual environment
```
python -m venv venv

source venv/bin/activate
```

Install dependencies:
```
pip install streamlit pypdf2 langchain python-dotenv faiss-cpu openai huggingface_hub pydantic==1.10.8
```

extra dependencies
```
pip install -U langchain-openai
pip install -U langchain-community faiss-cpu langchain-openai tiktoken
```
Note that you can also install faiss-gpu if you want to use the GPU enabled version

NOT SURE ABOUT THIS...
```
pip install langchain-community==0.0.11 pypdf==3.17.4 langchain==0.1.0 python-dotenv==1.0.0 langchain-openai==0.0.2.post1 faiss-cpu==1.7.4 tiktoken==0.5.2 langchainhub==0.1.14
```



to run the app:
```
streamlit run app.py
```

If you see the following error when uploading files
"AxiosError: Request failed with status code 403"
it means the required cookie is not loading properly. You must have this in your Application cookies of the browser:
_streamlit_xsrf
For this you might need to update your version of tornado
I suggest running 
pip freeze 
to check your version of Tordado is least v6.1 (which I didn’t) and then I ran 
run this command:
```
pip install -U tornado 
```
to upgrade to a more recent version.
After this clear the browser cache and do hard reload of your app, and make sure the correct _streamlit_xsrf cookie being saved in the browser cookie jar. 


Pydantic was added to solve issues with langchain running in python<3.9
```
pip install pydantic==1.10.8
```

