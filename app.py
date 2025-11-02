import os
import json
import time
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from flask_session import Session
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# استيراد الدوال الأساسية فقط
from GeneticFeatureSelector import (
    run_genetic_algorithm,
    load_and_prepare_data,
    run_feature_selection
)

# إعداد Flask
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # الحجم الأقصى 16MB
app.config["SESSION_TYPE"] = "filesystem"
app.secret_key = "secure-key-2025"
Session(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def allowed_file(filename):
    return filename.lower().endswith(".csv")

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename.strip() == "":
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
            preview = df.head(10).to_html(classes="table-container", index=False)
            info = {"filename": saved_name, "rows": df.shape[0], "cols": df.shape[1]}
            flash("تم رفع الملف بنجاح. يمكنك الآن تنفيذ المعالجة.", "success")
            return render_template("upload.html", preview=preview, info=info)
        except Exception as e:
            flash(f"خطأ أثناء قراءة الملف: {e}", "error")
            return redirect(request.url)
    return render_template("upload.html")

@app.route("/results")
def results():
    info = session.get("info")
    preview = session.get("preview")
    selected_preview = session.get("selected_preview")
    result_file = session.get("result_file")

    results = None
    if result_file and os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            results = json.load(f)

    return render_template("results.html", info=info, preview=preview,
                           selected_preview=selected_preview, results=results)

@app.route("/compare")
def compare():
    result_file = session.get("result_file")
    if not result_file or not os.path.exists(result_file):
        flash("لا توجد نتائج مقارنة لهذا الملف. يرجى تنفيذ المعالجة أولاً.", "warning")
        return redirect(url_for("results"))

    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {
        "features": data.get("features"),
        "accuracy": data.get("accuracy"),
        "time": data.get("time")
    }
    comparison = data.get("comparison", {})
    return render_template("compare.html", results=results, comparison=comparison)

@app.route("/history")
def history():
    folder = app.config["UPLOAD_FOLDER"]
    history = []

    for file in os.listdir(folder):
        if file.endswith("_result.json"):
            path = os.path.join(folder, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    history.append({
                        "filename": data.get("source_file", file.replace("_result.json", ".csv")),
                        "features": data.get("features"),
                        "accuracy": data.get("accuracy"),
                        "time": data.get("time"),
                        "target": data.get("target"),
                        "best_fitness": data.get("best_fitness"),
                        "best_generation": data.get("best_generation"),
                        "avg_fitness_final": data.get("avg_fitness_final")
                    })
            except Exception:
                continue

    history.sort(
        key=lambda x: os.path.getmtime(
            os.path.join(folder, f"{os.path.splitext(x['filename'])[0]}_result.json")
        ),
        reverse=True
    )

    return render_template("history.html", history=history)


@app.route("/compare/<filename>")
def compare_file(filename):
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(path):
        flash("ملف النتائج غير موجود.", "error")
        return redirect(url_for("history"))

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        flash(f"خطأ أثناء قراءة الملف: {e}", "error")
        return redirect(url_for("history"))

    results = {
        "features": data.get("features"),
        "accuracy": data.get("accuracy"),
        "time": data.get("time")
    }
    comparison = data.get("comparison", {})
    return render_template("compare.html", results=results, comparison=comparison)

@app.route("/docs")
def docs():
    return render_template("docs.html")

#  تنفيذ الخوارزمية 
@app.route("/process/<filename>")
def process_file(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(file_path):
        flash("الملف غير موجود.", "error")
        return redirect(url_for("upload"))

    # تشغيل الخوارزمية الجينية
    results = run_genetic_algorithm(file_path)
    target_col = results.get("target")
    X, y = load_and_prepare_data(file_path, target_col)

    if X is None or y is None:
        flash("فشل تحميل البيانات أو لم يُعثر على عمود هدف مناسب.", "error")
        return redirect(url_for("upload"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    base_model = RandomForestClassifier(n_estimators=10, random_state=42)
    comparison = run_feature_selection(X_train, X_test, y_train, y_test, base_model, results["features"])

    result_data = {
        "features": results.get("features"),
        "accuracy": results.get("accuracy"),
        "time": results.get("time"),
        "comparison": comparison,
        "selected_features": results.get("selected_features", []),
        "target": target_col,
        "source_file": filename,
        "best_fitness": results.get("best_fitness"),
        "best_generation": results.get("best_generation"),
        "avg_fitness_final": results.get("avg_fitness_final"),
        "plot": results.get("plot")
        
    }

    # حفظ النتائج
    result_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{os.path.splitext(filename)[0]}_result.json")
    save_json(result_path, result_data)

    # تحضير المعاينات
    df = pd.read_csv(file_path, encoding_errors="ignore")
    preview = df.head(10).to_html(classes="table-container", index=False)

    selected_features = results.get("selected_features")
    if selected_features:
        try:
            df_sel = df[selected_features + [target_col]]
            selected_preview = df_sel.head(10).to_html(classes="table-container", index=False)
        except Exception:
            selected_preview = "<p>تعذر عرض الميزات المختارة.</p>"
    else:
        selected_preview = "<p>لم يتم اختيار ميزات بواسطة GA.</p>"

    # تخزين الجلسة
    session["info"] = {"filename": filename, "rows": df.shape[0], "cols": df.shape[1]}
    session["preview"] = preview
    session["selected_preview"] = selected_preview
    session["result_file"] = result_path

    flash("تمت معالجة الملف بنجاح.", "success")
    return redirect(url_for("results"))

if __name__ == "__main__":
    app.run(debug=True)
