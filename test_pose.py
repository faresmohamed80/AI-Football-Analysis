import cv2
import os
from ultralytics import YOLO

# استخدام مسارات المشروع الحالية
from src.config import INPUT_VIDEO_PATH, BASE_DIR

def run_pose_experiment():
    print("⏳ جاري تحميل موديل المفاصل (YOLO-Pose)...")
    # تحميل الموديل (سيقوم بتحميله تلقائياً لو مش موجود)
    # نستخدم الحجم المتوسط (m) عشان نوازن بين السرعة والدقة في التجربة
    pose_model = YOLO("yolo26m-pose.pt")
    
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print("❌ لم يتم العثور على الفيديو.")
        return

    # إعدادات الحفظ (أول 300 فريم فقط للتجربة = حوالي 10-12 ثانية)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    
    out_path = os.path.join(BASE_DIR, "data", "output_data", "pose_experiment.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print("🚀 جاري تحليل حركة المفاصل...")
    frame_count = 0
    max_frames = 200 # هناخد 200 فريم بس عشان التجربة تخلص بسرعة

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # تشغيل موديل المفاصل مع تقليل نسبة الثقة ورفع جودة الصورة لتوسيع الاكتشاف
        results = pose_model(frame, imgsz=1280, conf=0.05, verbose=False)
        
        # رسم النتيجة المبدئية من Ultralytics (هيرسم الهيكل العظمي كامل)
        annotated_frame = results[0].plot()

        # استخراج المفاصل (Keypoints) لرسم تركيز خاص على "القدمين"
        if results[0].keypoints is not None and results[0].keypoints.data is not None:
            # keypoints shape: (num_persons, 17, 3) --> (x, y, confidence)
            keypoints = results[0].keypoints.data.cpu().numpy()
            
            for person in keypoints:
                # الكاحل الأيسر (Left Ankle) و الأيمن (Right Ankle) في COCO هم النقطتين 15 و 16
                if len(person) >= 17:
                    l_ankle = person[15]
                    r_ankle = person[16]
                    
                    # إذا كانت نسبة الثقة أعلى من 0.5، ارسم دائرة حمراء كبيرة حول الكاحل
                    if l_ankle[2] > 0.5:
                        cv2.circle(annotated_frame, (int(l_ankle[0]), int(l_ankle[1])), 8, (0, 0, 255), -1) # كاحل أيسر أحمر
                    if r_ankle[2] > 0.5:
                        cv2.circle(annotated_frame, (int(r_ankle[0]), int(r_ankle[1])), 8, (0, 0, 255), -1) # كاحل أيمن أحمر

        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % 20 == 0:
            print(f"تمت معالجة الفريم {frame_count}/{max_frames}")

    cap.release()
    out.release()
    print(f"✅ انتهت التجربة! الفيديو محفوظ في: {out_path}")
    print("لاحظ الدوائر الحمراء الكبيرة، دي بتمثل مكان القدم بالظبط اللي هنستخدمه لاكتشاف الركلة.")

if __name__ == "__main__":
    run_pose_experiment()
