from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.http import JsonResponse

from .image_to_text import convert
from .classifier import classify

import os

MEDIA_ROOT = settings.MEDIA_ROOT


def classify_poster(request):

    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        # uploaded_file_url = fs.url(filename)

        text = convert(os.path.join(MEDIA_ROOT, filename))
        print(text)

        classification = classify(text)
        print(classification)

        response = {
            'movie': text,
            'classification': classification
        }

        return JsonResponse(response)

    return render(request, 'upload.html')

