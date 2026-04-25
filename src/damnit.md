def run_ai_border_detection(self, image: np.ndarray) -> Dict:
    """Run TruScore AI border detection - optimized for custom 24-point centering system"""
    logger.info("✅ Running AI border detection (no centering calculations)")
    
    # Store original image dimensions (for size validation)
    orig_h, orig_w = image.shape[:2]
    logger.info(f"📱 Image size: {orig_w}x{orig_h} (Mobile device)")
    
    # Mobile-optimized confidence threshold (avoid over/under detection)
    confidence = max(0.35, min(0.5, getattr(self, 'detection_confidence', 0.4)))
    
    # Run YOLO detection (mobile-optimized settings)
    results = self.border_model(
        image, 
        conf=confidence,
        imgsz=640,  # Critical: Must match training size
        verbose=False,
        device='cpu'
    )
    
    # Initialize clean results structure for your centering system
    detected_borders = {
        'outer_border': None,
        'inner_border': None,
        'border_types': {},  # For API compatibility
        'confidence_scores': {'outer': 0.0, 'inner': 0.0},
        'raw_detections': [],
        'error': None
    }
    
    # No results case
    if not results or len(results) == 0:
        detected_borders['error'] = "No detections from model"
        return detected_borders
    
    # Process all valid detections
    all_detections = []
    
    for result in results:
        if not hasattr(result, 'boxes') or result.boxes is None:
            continue
            
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = float(confs[i])
            cls = int(classes[i])
            
            # VALIDATION: Card physical constraints (1530x2040 basis)
            width = x2 - x1
            height = y2 - y1
            
            # Outer border must be near full size (90-98% of image)
            if cls == 0 and (width < orig_w * 0.9 or height < orig_h * 0.9 or
                             width > orig_w or height > orig_h):
                continue
                
            # Inner border must be significantly smaller (70-85% of image)
            if cls == 1 and (width > orig_w * 0.85 or height > orig_h * 0.85 or
                             width < orig_w * 0.7 or height < orig_h * 0.7):
                continue
            
            all_detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf,
                'class_id': cls,
                'class_name': 'outer_border' if cls == 0 else 'inner_border'
            })
    
    # Sort by confidence and select highest for each class
    all_detections.sort(key=lambda x: x['confidence'], reverse=True)
    detected_borders['raw_detections'] = all_detections
    
    for detection in all_detections:
        cls = detection['class_id']
        if cls == 0 and detected_borders['confidence_scores']['outer'] < detection['confidence']:
            detected_borders['outer_border'] = np.array(detection['bbox'])
            detected_borders['confidence_scores']['outer'] = detection['confidence']
        elif cls == 1 and detected_borders['confidence_scores']['inner'] < detection['confidence']:
            detected_borders['inner_border'] = np.array(detection['bbox'])
            detected_borders['confidence_scores']['inner'] = detection['confidence']
    
    # Final validation for your 24-point system
    if detected_borders['outer_border'] is None:
        detected_borders['error'] = "Outer border missing"
    elif detected_borders['inner_border'] is None:
        detected_borders['error'] = "Inner border missing"
    
    if detected_borders['error']:
        logger.error(f"❌ {detected_borders['error']} (outer conf: {detected_borders['confidence_scores']['outer']:.2f}, inner: {detected_borders['confidence_scores']['inner']:.2f})")
    else:
        logger.info(f"✅ Borders detected: outer={detected_borders['confidence_scores']['outer']:.2f}, inner={detected_borders['confidence_scores']['inner']:.2f}")
    
    return detected_borders
