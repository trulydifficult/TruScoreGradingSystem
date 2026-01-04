from centering_analyzer import CenteringAnalyzer, format_results_text, yolo_box_to_polygon

# If you currently have YOLO boxes:
outer_poly = yolo_box_to_polygon(outer_box)   # (x1, y1, x2, y2)
inner_poly = yolo_box_to_polygon(inner_box)

an = CenteringAnalyzer(image_path, outer_poly, inner_poly)
results = an.run_analysis(show_visual=True)

# Show overlay
label.setPixmap(results.pixmap)  # any QLabel in your tab

# Show text report
text_edit.setPlainText(format_results_text(results))

# Or access pieces directly:
results.measurements_mm         # 24 values, ordered Top(1–5), Bottom(6–10), Left(11–17), Right(18–24)
results.groups['top']['avg']    # etc.
results.ratios['top_bottom']    # (e.g., (55.0, 45.0))
results.verdict                 # final explanation string
