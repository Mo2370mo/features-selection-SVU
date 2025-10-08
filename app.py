import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd

# إعداد Flask
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # الحجم الأقصى 16MB
app.secret_key = "change-me" 

# تأكد أن مجلد الرفع موجود
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# التحقق من الامتداد
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "csv"

# الصفحة الرئيسية
@app.route("/")
def index():
    return render_template("index.html")

# رفع الملفات
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("الرجاء اختيار ملف CSV.", "error")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("يسمح فقط بملفات CSV.", "error")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        saved_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        path = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
        file.save(path)

        try:
            df = pd.read_csv(path, encoding_errors="ignore")
        except Exception as e:
            flash(f"خطأ في قراءة الملف: {e}", "error")
            return redirect(request.url)
        
        flash("تم رفع الملف بنجاح.", "success")

        preview = df.to_html(classes="table-container", index=False)
        info = {"filename": saved_name, "rows": df.shape[0], "cols": df.shape[1]}
        return render_template("results.html", preview=preview, info=info)

    return render_template("upload.html")

# صفحة النتائج
@app.route("/results")
def results():
    return render_template("results.html", preview=None, info=None)

# صفحة المقارنة
@app.route("/compare")
def compare():
    return render_template("compare.html")

# صفحة التوثيق
@app.route("/docs")
def docs():
    return render_template("docs.html")

# تنفيذ الخوارزمية  
@app.route("/process/<filename>")
def process_file(filename):
 
    flash("تم إرسال الملف للمعالجة .")
    return redirect(url_for("results"))

# تشغيل التطبيق
if __name__ == "__main__":
    app.run(debug=True)
