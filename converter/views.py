from django.shortcuts import render, redirect
from .models import Image
from .forms import ImageForm
from .main import pipeline

def upload(request):
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            # フォームを保存
            # form.save()
            # フォームから画像を取得
            image = form.cleaned_data['image']
            # 画像変換
            file_name, file_name_2 = pipeline(image)

            return render(request, 'converter/upload.html', {'file_name': file_name, 'file_name_2': file_name_2})
    else:
        form = ImageForm()

    context = {'form':form}
    return render(request, 'converter/upload.html', context)

def showall(request):
    images = Image.objects.all()
    context = {'images': images}

    return render(request, 'converter/showall.html', context)
