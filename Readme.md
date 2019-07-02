# Poster Upload

## Install
```
pip install -r posterclassifier/requirements.txt
```

# Running

´´´
python3 posterclassifier/manage.py runserver
´´´

## Usage

Open `http://127.0.0.1:8000/classify/poster` in your browser

Output
```
{
'movie': {Extracted text form poster}
'classification': {Predicted rating}
}
```
