from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

name = default_storage.save(
    "test/hello.txt",
    ContentFile(b"VAS Django S3 test"),
)

print(name)
print(default_storage.exists(name))