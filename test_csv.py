import pandas as pd
pd.DataFrame({
    "yorum": [
        "Ürün çok kaliteliydi, kesinlikle tavsiye ederim",
        "Berbat bir deneyimdi, bir daha almam",
        "İdare eder, ne iyi ne kötü",
        "Hızlı kargo, güzel paketleme teşekkürler",
        "Para vermekten pişman oldum tamamen",
    ]
}).to_csv("test.csv", index=False, encoding="utf-8")
print("test.csv oluşturuldu")