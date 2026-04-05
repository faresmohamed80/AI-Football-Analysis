from collections import Counter

class NumberVotingSystem:
    def __init__(self, required_frames=30):
        """
        نظام لتجميع قراءات الأرقام واختيار الأكثر تكراراً
        required_frames: عدد الفريمات (أو القراءات) المطلوبة لأخذ قرار نهائي
        """
        self.required_frames = required_frames
        
        # قاموس (Dictionary) لتخزين كل القراءات لكل لاعب
        # الشكل: {track_id: ["10", "10", "7", "10"]}
        self.history = {} 
        
        # قاموس لتخزين الرقم النهائي بعد ما ينجح في التصويت
        # الشكل: {track_id: "10"}
        self.final_numbers = {} 

    def update(self, track_id, predicted_number):
        """
        يستقبل الـ ID المؤقت بتاع اللاعب من الـ Tracker، والرقم اللي الموديل قرأه في الفريم ده
        """
        # 1. لو اللاعب ده إحنا أصلاً ثبتنا رقمه قبل كده، رجع الرقم النهائي فوراً
        if track_id in self.final_numbers:
            return self.final_numbers[track_id]

        # 2. لو الرقم لسه ماتثبتش، نبدأ نخزن القراءات
        if track_id not in self.history:
            self.history[track_id] = []
        
        # لو الموديل قرأ رقم فعلاً (مش None)، ضيفه للتاريخ بتاع اللاعب
        if predicted_number is not None and predicted_number != "":
            self.history[track_id].append(predicted_number)

        # 3. هل جمعنا قراءات كفاية (مثلاً 30 قراءة)؟
        if len(self.history[track_id]) >= self.required_frames:
            # احسب الرقم الأكثر تكراراً (Voting)
            most_common = Counter(self.history[track_id]).most_common(1)[0][0]
            
            # احفظه كرقم نهائي للاعب ده عشان مانحسبوش تاني
            self.final_numbers[track_id] = most_common
            
            # ممكن تفضي الـ history بتاع اللاعب ده عشان توفر مساحة في الرامات
            del self.history[track_id] 
            
            return most_common

        # لو لسه مكملناش الـ 30 قراءة، هنرجع None أو نكتب "جاري التعرف..."
        return "Loading..."