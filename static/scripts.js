window.addEventListener("DOMContentLoaded", () => {
  // عرض اسم الملف عند الاختيار
  const fileInput = document.querySelector('input[type="file"]');
  const fileNameDisplay = document.getElementById("file-name");
  if (fileInput && fileNameDisplay) {
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0) {
        fileNameDisplay.textContent = "الملف المحدد: " + fileInput.files[0].name;
      } else {
        fileNameDisplay.textContent = "";
      }
    });
  }

  // تعطيل زر الرفع أثناء المعالجة
  const form = document.querySelector("form");
  if (form) {
    form.addEventListener("submit", () => {
      const btn = form.querySelector("button[type='submit']");
      if (btn) {
        btn.disabled = true;
        btn.textContent = "جاري الرفع...";
      }
    });
  }

  // تمرير ناعم لأعلى عند تحميل الصفحة
  window.scrollTo({ top: 0, behavior: "smooth" });

  // تثبيت رؤوس الجداول أثناء التمرير
  const tables = document.querySelectorAll(".comparison-table");
  tables.forEach(tbl => {
    const thead = tbl.querySelector("thead");
    if (thead) {
      thead.style.position = "sticky";
      thead.style.top = "0";
      thead.style.backgroundColor = "#0078d4";
    }
  });
});
