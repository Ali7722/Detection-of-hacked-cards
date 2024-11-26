import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# التحقق من اسم الملف
file_name = 'payment_fraud.csv'

# تحقق من وجود الملف في نفس المجلد
if not os.path.exists(file_name):
    print(f"Error: File '{file_name}' not found in the current directory.")
    print("Please make sure the file is in the same directory as the script.")
    exit()
# قراءة الملف
df = pd.read_csv(file_name)

# -------------------- تحليل البيانات --------------------
print("معلومات البيانات:")
print(df.info())
print("\nملخص البيانات:")
print(df.describe())

# التحقق من القيم المفقودة
missing_values = df.isnull().sum()
print("\nالقيم المفقودة في كل عمود:")
print(missing_values)

# رسم العلاقة بين المتغيرات
sns.pairplot(df, hue='label')
plt.show()

# -------------------- معالجة البيانات --------------------
# تعويض القيم المفقودة (إن وجدت)
df.fillna(0, inplace=True)

# معالجة عمود paymentMethodAgeDays
df['paymentMethodAgeDays'] = df['paymentMethodAgeDays'].replace(0, df['paymentMethodAgeDays'].median())

# تحويل الأعمدة النصية إلى ترميز عددي
df = pd.get_dummies(df, columns=['paymentMethod'])

# فصل البيانات إلى ميزات ومستهدف
X = df.drop('label', axis=1)
y = df['label']

# -------------------- تقسيم البيانات --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=17)

# -------------------- تدريب النموذج --------------------
clf = RandomForestClassifier(random_state=17)
clf.fit(X_train, y_train)

# التنبؤ باستخدام مجموعة الاختبار
y_pred = clf.predict(X_test)

# -------------------- تحليل النتائج --------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------- فرز النتائج --------------------
# إضافة التوقعات إلى مجموعة الاختبار
X_test['prediction'] = y_pred
X_test['label'] = y_test

# تصفية البطاقات الآمنة والمخترقة
safe_cards = X_test[X_test['prediction'] == 0]
fraudulent_cards = X_test[X_test['prediction'] == 1]

# حفظ النتائج إلى ملفات نصية
safe_cards[['accountAgeDays', 'numItems', 'paymentMethodAgeDays']].to_csv('safe_cards.txt', index=False, header=True)
fraudulent_cards[['accountAgeDays', 'numItems', 'paymentMethodAgeDays']].to_csv('fraudulent_cards.txt', index=False, header=True)

print("\nتم إنشاء ملفات النتائج: 'safe_cards.txt' و 'fraudulent_cards.txt'")
