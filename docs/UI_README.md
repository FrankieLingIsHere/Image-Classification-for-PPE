# 🦺 PPE Safety Detection Web Interface

A user-friendly web interface for analyzing Personal Protective Equipment (PPE) compliance in workplace images using AI-powered detection.

## 🚀 Quick Start

### Method 1: Using Launcher Scripts

**Windows:**
```bash
# Double-click or run:
launch_ui.bat
```

**Linux/Mac:**
```bash
# Make executable and run:
chmod +x launch_ui.sh
./launch_ui.sh
```

### Method 2: Manual Launch

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Launch Streamlit
streamlit run streamlit_app.py --server.port 8501
```

### Method 3: Direct Python Command

```bash
# From project root directory
.venv\Scripts\python -m streamlit run streamlit_app.py
```

## 🌐 Accessing the Interface

Once launched, open your web browser and go to:
- **URL:** http://localhost:8501
- The interface will automatically open in your default browser

## 📱 Using the Interface

### 1. **Upload an Image**
- Click "Choose an image file" 
- Select PNG, JPG, or JPEG files
- Construction site or workplace images work best

### 2. **Configure Settings** (Sidebar)
- **Confidence Threshold:** Adjust detection sensitivity (0.1-0.9)
- **Show Bounding Boxes:** Toggle visual detection overlays
- **Show Detailed Analysis:** Enable/disable technical details

### 3. **View Results**
The interface provides:
- **📊 Key Metrics:** People count, PPE items, violations
- **🚨 Compliance Status:** Overall safety assessment
- **🦺 Safety Summary:** Detailed safety analysis
- **🎯 Scene Description:** AI-generated image description
- **📈 Detection Breakdown:** Interactive charts and graphs
- **🔍 Detailed Detections:** Technical detection data

### 4. **Export Results**
- Click "Download Analysis Report (JSON)"
- Get comprehensive analysis data for documentation
- Timestamped reports for record-keeping

## 🛠️ Features

### PPE Detection Categories

**✅ Safety Equipment Detected:**
- Hard hats
- Safety vests
- Safety gloves  
- Safety boots
- Eye protection

**❌ Violations Detected:**
- Missing hard hats
- Missing safety vests
- Missing safety gloves
- Missing safety boots
- Missing eye protection

### Analysis Capabilities
- **Real-time Detection:** Instant PPE analysis
- **Visual Annotations:** Bounding box overlays
- **Compliance Scoring:** Automatic safety assessment
- **Interactive Charts:** Detection breakdown visualizations
- **Export Functionality:** JSON reports for documentation

## 📊 Understanding Results

### Compliance Status
- **✅ COMPLIANT:** All workers properly equipped
- **❌ CRITICAL NON-COMPLIANCE:** Violations detected

### Detection Confidence
- **High (0.7-1.0):** Very reliable detection
- **Medium (0.4-0.7):** Good detection quality  
- **Low (0.1-0.4):** Less reliable, review manually

### Bounding Box Colors
- **🟡 Yellow:** Person detection
- **🟢 Green:** PPE equipment (good)
- **🔴 Red:** PPE violations (missing equipment)

## 🔧 Troubleshooting

### Common Issues

**1. Model Loading Errors**
```
❌ Failed to load model: [error message]
```
- Ensure `models/best_model_regularized.pth` exists
- Check that all dependencies are installed
- Verify virtual environment is activated

**2. Image Upload Issues**
- Supported formats: PNG, JPG, JPEG
- Maximum file size: ~10MB recommended
- Check file permissions

**3. Performance Issues**
- First run may be slow (model loading)
- Large images take longer to process
- CPU inference is slower than GPU

**4. Port Already in Use**
```
OSError: [Errno 98] Address already in use
```
- Try a different port: `streamlit run streamlit_app.py --server.port 8502`
- Or stop existing Streamlit processes

### Performance Tips

**For Faster Processing:**
- Resize very large images before upload
- Use confidence threshold ≥ 0.3 for cleaner results
- Disable bounding boxes for faster display

**For Better Accuracy:**
- Use clear, well-lit images
- Ensure workers are clearly visible
- Avoid heavily cropped or blurry images

## 📁 Project Structure

```
├── streamlit_app.py          # Main UI application
├── launch_ui.bat            # Windows launcher
├── launch_ui.sh             # Linux/Mac launcher
├── src/models/              # AI models
│   └── hybrid_ppe_model.py  # PPE detection model
├── models/                  # Trained model files
│   └── best_model_regularized.pth
└── requirements.txt         # Dependencies
```

## 🎯 Use Cases

### Safety Inspections
- Construction site audits
- Workplace safety assessments
- Compliance monitoring
- Training documentation

### Industrial Applications
- Manufacturing safety
- Oil & gas operations
- Mining operations
- Chemical plants

### Educational Use
- Safety training programs
- Compliance education
- Risk assessment training
- Academic research

## 📞 Support

For technical issues or questions:
1. Check this README
2. Review terminal output for error messages
3. Ensure all dependencies are properly installed
4. Verify model files are in correct locations

## 🔄 Updates

The interface supports the latest hybrid PPE detection model with:
- Improved coordinate decoding
- Better violation detection
- Enhanced classification accuracy
- Real-time analysis capabilities

---

**🦺 Stay Safe! Proper PPE saves lives!** 🦺