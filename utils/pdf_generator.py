"""
PDF report generation utilities
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import os
import numpy as np
from datetime import datetime
from config import Config


def generate_pdf_report(df, training_results=None, evaluation_results=None):
    """
    Generate a comprehensive PDF report
    
    Args:
        df: DataFrame with the dataset
        training_results: Results from model training
        evaluation_results: Results from model evaluation
    
    Returns:
        Path to generated PDF file
    """
    os.makedirs(Config.REPORTS_FOLDER, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'obesity_prediction_report_{timestamp}.pdf'
    filepath = os.path.join(Config.REPORTS_FOLDER, filename)
    
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    temp_paths = []  # Collect temp files for ROC images to delete after build
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a5490'),
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Obesity Risk Prediction Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Date
    date_style = ParagraphStyle(
        'CustomDate',
        parent=styles['Normal'],
        fontSize=10,
        alignment=1
    )
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Dataset Information
    story.append(Paragraph("Dataset Information", styles['Heading2']))
    dataset_info = [
        ['Total Samples', str(len(df))],
        ['Total Features', str(len(df.columns))],
        ['Numerical Features', str(len(df.select_dtypes(include=[np.number]).columns))],
        ['Categorical Features', str(len(df.select_dtypes(include=['object']).columns))]
    ]
    table = Table(dataset_info, colWidths=[3*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3*inch))
    
    # Model Training Results
    if training_results:
        story.append(Paragraph("Model Training Results", styles['Heading2']))
        training_data = [['Model', 'Accuracy', 'Status']]
        for result in training_results:
            training_data.append([
                result.get('model', 'N/A'),
                f"{result.get('accuracy', 0):.4f}" if 'accuracy' in result else 'N/A',
                result.get('status', 'N/A')
            ])
        
        table = Table(training_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
    
    # Model Evaluation Results
    best_model_name = None
    best_model_metrics = None
    if evaluation_results:
        story.append(Paragraph("Model Evaluation Metrics", styles['Heading2']))
        eval_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
        for model_name, result in evaluation_results.items():
            if 'error' not in result:
                acc = float(result.get('accuracy', 0))
                eval_data.append([
                    model_name.replace('_', ' ').title(),
                    f"{acc:.4f}",
                    f"{float(result.get('precision', 0)):.4f}",
                    f"{float(result.get('recall', 0)):.4f}",
                    f"{float(result.get('f1_score', 0)):.4f}",
                    f"{result.get('roc_auc', 'N/A')}" if isinstance(result.get('roc_auc'), str) else f"{float(result.get('roc_auc', 0)):.4f}"
                ])
                if best_model_metrics is None or acc > best_model_metrics.get('accuracy', 0):
                    best_model_name = model_name.replace('_', ' ').title()
                    best_model_metrics = result
        
        if len(eval_data) > 1:
            table = Table(eval_data, colWidths=[1.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 0.3*inch))
            
            # Add ROC Curves to PDF
            story.append(Paragraph("ROC-AUC Analysis", styles['Heading2']))
            import base64
            from reportlab.platypus import Image as RLImage
            import tempfile
            
            temp_paths = []
            for model_name, result in evaluation_results.items():
                if 'roc_curve_image' in result and result['roc_curve_image']:
                    story.append(Paragraph(f"ROC Curve: {model_name.replace('_', ' ').title()}", styles['Heading3']))
                    # Save base64 to temp file
                    img_data = base64.b64decode(result['roc_curve_image'])
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        tmp_file.write(img_data)
                        tmp_path = tmp_file.name
                    temp_paths.append(tmp_path)
                    # Add to PDF
                    story.append(RLImage(tmp_path, width=5*inch, height=4*inch))
                    story.append(Spacer(1, 0.2*inch))

    # Target distribution (how many patients per obesity category)
    if 'Obesity Category' in df.columns:
        story.append(Paragraph("Obesity Category Distribution", styles['Heading2']))
        counts = df['Obesity Category'].value_counts().to_dict()
        dist_data = [['Category', 'Number of Individuals']]
        for k, v in counts.items():
            dist_data.append([str(k), str(v)])
        dist_table = Table(dist_data, colWidths=[3*inch, 3*inch])
        dist_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(dist_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Conclusion
    story.append(Paragraph("Conclusion & Business Interpretation", styles['Heading2']))
    if best_model_name and best_model_metrics:
        conclusion_text = f"""
        The best performing model on this dataset is <b>{best_model_name}</b> with an accuracy of
        <b>{best_model_metrics.get('accuracy', 0):.2%}</b> and an F1‑score of
        <b>{best_model_metrics.get('f1_score', 0):.2%}</b>. This model should be used for production
        predictions and decision support.

        The predicted obesity category for new individuals should be interpreted as a <i>risk indicator</i>
        rather than a clinical diagnosis. It is recommended that clinicians review model outputs together
        with BMI, lifestyle information and routine laboratory results before taking action.
        """
    else:
        conclusion_text = """
        This report contains the analysis and model training results for the obesity risk prediction dataset.
        Multiple machine learning models have been trained and evaluated to predict obesity categories
        based on various health and lifestyle factors.
        """
    story.append(Paragraph(conclusion_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    # Clean up temp files used for ROC images (Windows-safe)
    for tmp_path in temp_paths:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
    return filepath

