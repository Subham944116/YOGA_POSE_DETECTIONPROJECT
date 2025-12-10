import base64
import io
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from PIL import Image
import numpy as np
import math
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    mp = None
    MP_AVAILABLE = False

POSE_MAP = {
    'tree': {
        'name': 'Tree Pose (Vrikshasana)',
        'description': 'Balance on one leg, opposite foot on inner thigh, hands above head. Benefits: improves balance, strengthens legs, enhances concentration.'
    },
    'warrior_ii': {
        'name': 'Warrior II (Virabhadrasana II)',
        'description': 'Lunge with arms extended, gaze over front hand. Benefits: strengthens legs and core, increases stamina, opens hips and chest.'
    },
    'downward_dog': {
        'name': 'Downward-Facing Dog (Adho Mukha Svanasana)',
        'description': 'Hands and feet on ground, hips raised forming an inverted V. Benefits: stretches spine, strengthens shoulders, improves blood circulation.'
    },
    'trikonasana': {
        'name': 'Triangle Pose (Trikonasana)',
        'description': 'Feet apart, one hand near front foot, other hand up; torso stretched sideways. Benefits: improves flexibility, strengthens legs, enhances digestion.'
    },
    'natarajasana': {
        'name': 'Dancer Pose (Natarajasana)',
        'description': 'Standing balance, one foot held behind by hand; chest lifted. Benefits: boosts balance, strengthens back/legs, opens chest and shoulders.'
    },
    'bhujangasana': {
        'name': 'Cobra Pose (Bhujangasana)',
        'description': 'Lying face down, hands under shoulders, chest lifted by spine extension. Benefits: strengthens spine, reduces back stiffness, opens chest and lungs.'
    },
    'bridge': {
        'name': 'Bridge Pose (Setu Bandha Sarvangasana)',
        'description': 'Backbend with hips lifted. Benefits: strengthens glutes and spine, reduces stress, improves digestion.'
    },
    'paschimottanasana': {
        'name': 'Seated Forward Bend (Paschimottanasana)',
        'description': 'Seated, torso bending forward toward feet. Benefits: stretches hamstrings, calms the mind, improves flexibility.'
    }
}

def info(request):
    return render(request, "yoga_app/info.html")
def index(request):
    return render(request, 'yoga_app/index.html')

def _dist(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (getattr(a, 'z', 0) - getattr(b, 'z', 0))**2)

def detect_pose(request):
    if request.method != 'POST':
        return HttpResponseBadRequest('Only POST allowed')

    if not MP_AVAILABLE:
        return JsonResponse({'error': 'mediapipe not installed. Install mediapipe and restart the server.'}, status=500)

    data_url = request.POST.get('image')
    if not data_url:
        return HttpResponseBadRequest('No image data received')

    
    stream_mode = request.POST.get('stream', '0') == '1'

    
    try:
        header, encoded = data_url.split(',', 1)
        img_data = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_np = np.array(img)[:, :, ::-1] 
        img_rgb = np.array(img)  
    except Exception as e:
        return JsonResponse({'error': f'Invalid image data: {e}'}, status=400)

    # Run MediaPipe pose detection
    mp_pose = mp.solutions.pose
    # static_image_mode False is recommended for camera/streaming; True for single images.
    with mp_pose.Pose(static_image_mode=not stream_mode, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(img_rgb)
        if not results.pose_landmarks:
            return JsonResponse({'pose': None, 'landmarks': None, 'message': 'No person/pose detected'})

        lm = results.pose_landmarks.landmark
        # handy shorthand
        L = mp_pose.PoseLandmark
        # fetch landmarks we will use (safe access)
        def get(idx):
            return lm[idx.value]

        left_ankle = get(L.LEFT_ANKLE)
        right_ankle = get(L.RIGHT_ANKLE)
        left_wrist = get(L.LEFT_WRIST)
        right_wrist = get(L.RIGHT_WRIST)
        left_shoulder = get(L.LEFT_SHOULDER)
        right_shoulder = get(L.RIGHT_SHOULDER)
        left_hip = get(L.LEFT_HIP)
        right_hip = get(L.RIGHT_HIP)
        nose = get(L.NOSE)
        left_knee = get(L.LEFT_KNEE)
        right_knee = get(L.RIGHT_KNEE)
        left_elbow = get(L.LEFT_ELBOW)
        right_elbow = get(L.RIGHT_ELBOW)

        # compute some basic derived values
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        avg_hip_y = (left_hip.y + right_hip.y) / 2.0

        # Naive heuristics (ordered). These are approximate rules-of-thumb only.
        pose_guess = None

        # 1) Dancer Pose (Natarajasana):
        # one ankle clearly higher than the other AND same-side wrist close to that ankle
        if abs(left_ankle.y - right_ankle.y) > 0.12:
            # detect which foot is lifted
            if left_ankle.y < right_ankle.y:  # left foot raised (smaller y = higher in image)
                if _dist(left_ankle, left_wrist) < 0.12:
                    pose_guess = 'natarajasana'
            else:
                if _dist(right_ankle, right_wrist) < 0.12:
                    pose_guess = 'natarajasana'

        # 2) Triangle Pose (Trikonasana): one arm up, one arm down and torso sideways
        if pose_guess is None:
            # check large vertical separation between wrists and asymmetry between shoulders
            wrists_vert_diff = abs(left_wrist.y - right_wrist.y)
            shoulders_x_diff = abs(left_shoulder.x - right_shoulder.x)
            # One wrist much higher than the other and the vertical span is wide
            if wrists_vert_diff > 0.18 and abs(left_wrist.x - right_wrist.x) > 0.25:
                # also require torso is extended (shoulder-hip vertical asymmetry)
                if abs(left_shoulder.y - right_shoulder.y) > 0.03:
                    pose_guess = 'trikonasana'

        # 3) Seated Forward Bend (Paschimottanasana):
        # both wrists close to ankles (hands reaching feet)
        if pose_guess is None:
            d_l = _dist(left_wrist, left_ankle)
            d_r = _dist(right_wrist, right_ankle)
            if d_l < 0.14 and d_r < 0.14:
                pose_guess = 'paschimottanasana'

        # 4) Bridge Pose: hips lifted above the level of shoulders (hip y < shoulder y)
        if pose_guess is None:
            if avg_hip_y < avg_shoulder_y - 0.05:
                pose_guess = 'bridge'

        # 5) Cobra Pose (Bhujangasana): chest/head noticeably lifted (nose higher relative to shoulders)
        if pose_guess is None:
            if nose.y < avg_shoulder_y - 0.03:
                # and wrists are near shoulders (hands under chest)
                if _dist(left_wrist, left_shoulder) < 0.18 and _dist(right_wrist, right_shoulder) < 0.18:
                    pose_guess = 'bhujangasana'

        # 6) Tree Pose (existing): one ankle much higher than the other, but without hand-to-foot contact like dancer
        if pose_guess is None:
            if abs(left_ankle.y - right_ankle.y) > 0.12:
                # ensure wrist-to-ankle distance is not small (so not dancer)
                if _dist(left_ankle, left_wrist) > 0.14 and _dist(right_ankle, right_wrist) > 0.14:
                    pose_guess = 'tree'

        # 7) Warrior II (as before): wrists wide apart and roughly at shoulder height
        if pose_guess is None:
            if abs(left_wrist.y - right_wrist.y) > 0.20 and abs(left_wrist.y - avg_shoulder_y) < 0.20 and abs(right_wrist.y - avg_shoulder_y) < 0.20:
                pose_guess = 'warrior_ii'

        # Fallback to Downward Dog
        if pose_guess is None:
            pose_guess = 'downward_dog'

        info = POSE_MAP.get(pose_guess, {'name': pose_guess, 'description': ''})
        # Respond with pose info and landmarks
        return JsonResponse({
            'pose': pose_guess,
            'pose_name': info['name'],
            'description': info['description'],
            'landmarks': [
                {
                    'idx': i,
                    'x': float(l.x),
                    'y': float(l.y),
                    'z': float(getattr(l, 'z', 0.0)),
                    'visibility': float(getattr(l, 'visibility', 0.0))
                }
                for i, l in enumerate(results.pose_landmarks.landmark)
            ]
        })



import base64
import io
import math
import json
from PIL import Image
import numpy as np

from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt

# MediaPipe import
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    # Define short aliases for landmarks for readability in config below
    LM = mp_pose.PoseLandmark
except Exception as e:
    mp_pose = None
    LM = None
    print("MediaPipe import error:", e)


# --- Utility Functions ---

def angle_between_points(a, b, c):
    """ compute angle between three points (in degrees) at vertex b """
    # a, b, c are iterables (x, y)
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    # handle zero-length
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0.0
    # Calculate cosine angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Ensure value is within valid domain for arccos due to tiny floating point errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return float(angle)

# --- Configuration ---

# 1. TARGET_POSES
# Defines WHICH joints define the pose.
# Keys: joint name -> tuple of (Point A, Vertex Point B, Point C) using indices.
TARGET_POSES = {
    "warrior_ii": {
        # Assuming Left leg forward bent, Right leg back straight
        "front_knee": (LM.LEFT_HIP.value, LM.LEFT_KNEE.value, LM.LEFT_ANKLE.value),
        "back_knee": (LM.RIGHT_HIP.value, LM.RIGHT_KNEE.value, LM.RIGHT_ANKLE.value),
        # Arms parallel to floor. Angle at shoulder formed by hip and elbow.
        "left_arm_level": (LM.LEFT_ELBOW.value, LM.LEFT_SHOULDER.value, LM.LEFT_HIP.value),
        "right_arm_level": (LM.RIGHT_ELBOW.value, LM.RIGHT_SHOULDER.value, LM.RIGHT_HIP.value),
    },
    "tree": {
        # Assuming Right leg is bent and placed on inner left thigh
        "standing_leg_knee": (LM.LEFT_HIP.value, LM.LEFT_KNEE.value, LM.LEFT_ANKLE.value),
        "bent_leg_knee": (LM.RIGHT_HIP.value, LM.RIGHT_KNEE.value, LM.RIGHT_ANKLE.value),
        # Check if hips are level horizontally
        "hip_levelness": (LM.LEFT_SHOULDER.value, LM.LEFT_HIP.value, LM.RIGHT_HIP.value),
    },
    "downward_dog": {
        "left_knee": (LM.LEFT_HIP.value, LM.LEFT_KNEE.value, LM.LEFT_ANKLE.value),
        "right_knee": (LM.RIGHT_HIP.value, LM.RIGHT_KNEE.value, LM.RIGHT_ANKLE.value),
        "left_elbow": (LM.LEFT_SHOULDER.value, LM.LEFT_ELBOW.value, LM.LEFT_WRIST.value),
        "right_elbow": (LM.RIGHT_SHOULDER.value, LM.RIGHT_ELBOW.value, LM.RIGHT_WRIST.value),
        # The characteristic inverted 'V' shape at the hip
        "hip_apex": (LM.LEFT_SHOULDER.value, LM.LEFT_HIP.value, LM.LEFT_KNEE.value),
    },
    "plank": {
        # Arms straight
        "left_elbow": (LM.LEFT_SHOULDER.value, LM.LEFT_ELBOW.value, LM.LEFT_WRIST.value),
        "right_elbow": (LM.RIGHT_SHOULDER.value, LM.RIGHT_ELBOW.value, LM.RIGHT_WRIST.value),
        # The crucial straight line of the body: Shoulder - Hip - Ankle
        "body_alignment_l": (LM.LEFT_SHOULDER.value, LM.LEFT_HIP.value, LM.LEFT_ANKLE.value),
    },
    "triangle": {
         # Assuming bending towards the Right side
        "front_leg_knee": (LM.RIGHT_HIP.value, LM.RIGHT_KNEE.value, LM.RIGHT_ANKLE.value),
        "back_leg_knee": (LM.LEFT_HIP.value, LM.LEFT_KNEE.value, LM.LEFT_ANKLE.value),
        # Arms forming a straight line (180) across shoulders
        "arms_t_shape": (LM.RIGHT_ELBOW.value, LM.RIGHT_SHOULDER.value, LM.LEFT_SHOULDER.value),
        # Side bend angle at the hip
        "torso_bend": (LM.LEFT_SHOULDER.value, LM.LEFT_HIP.value, LM.LEFT_KNEE.value)
    },
    "mountain": {
        "left_knee": (LM.LEFT_HIP.value, LM.LEFT_KNEE.value, LM.LEFT_ANKLE.value),
        "right_knee": (LM.RIGHT_HIP.value, LM.RIGHT_KNEE.value, LM.RIGHT_ANKLE.value),
        # Vertical Posture: Line from ankle to hip to shoulder should be roughly straight
        "posture_line_l": (LM.LEFT_SHOULDER.value, LM.LEFT_HIP.value, LM.LEFT_ANKLE.value),
        # Arms straight at sides
        "left_elbow": (LM.LEFT_SHOULDER.value, LM.LEFT_ELBOW.value, LM.LEFT_WRIST.value),
    },
     "cobra": {
        # Lying down, legs straight
        "left_knee": (LM.LEFT_HIP.value, LM.LEFT_KNEE.value, LM.LEFT_ANKLE.value),
        # Back bend: Angle formed by Knee-Hip-Shoulder
        "back_extension": (LM.LEFT_KNEE.value, LM.LEFT_HIP.value, LM.LEFT_SHOULDER.value),
        # Elbows bent backward usually around 90-120 depending on height
        "left_elbow_bend": (LM.LEFT_SHOULDER.value, LM.LEFT_ELBOW.value, LM.LEFT_WRIST.value),
    }
}

# 2. POSE_ANGLE_TARGETS
# Defines the ideal angles in degrees.
# IMPORTANT: These are theoretical placeholders. Calibrate with real data.
POSE_ANGLE_TARGETS = {
    "warrior_ii": {
        "front_knee": 90.0,  # Ideal 90 degree bend
        "back_knee": 180.0,  # Straight leg
        "left_arm_level": 90.0, # Arm out 90 deg from torso line
        "right_arm_level": 90.0,
    },
    "tree": {
        "standing_leg_knee": 180.0, # Straight
        # Bent leg is acute, depends on flexibility, approx 60 deg
        "bent_leg_knee": 60.0,
        # The angle between shoulder-hip-hip should be roughly 90 if standing straight up
        "hip_levelness": 90.0,
    },
     "downward_dog": {
        "left_knee": 180.0, # Straight legs
        "right_knee": 180.0,
        "left_elbow": 180.0, # Straight arms
        "right_elbow": 180.0,
        # The hips form a sharp angle, usually between 70-90 degrees
        "hip_apex": 80.0,
    },
     "plank": {
        "left_elbow": 180.0, # Arms locked straight (high plank)
        "right_elbow": 180.0,
        # Body should be a perfectly straight line
        "body_alignment_l": 180.0,
    },
    "triangle": {
        "front_leg_knee": 180.0, # Both legs straight
        "back_leg_knee": 180.0,
        # Arms forming a continuous straight line from fingertip to fingertip
        "arms_t_shape": 180.0,
        # Torso bent sideways relative to the leg line. Approx 130 deg.
        "torso_bend": 130.0,
    },
    "mountain": {
        "left_knee": 180.0,
        "right_knee": 180.0,
        "posture_line_l": 180.0, # Standing perfectly straight
        "left_elbow": 180.0,
    },
     "cobra": {
        "left_knee": 180.0, # Legs straight on floor
        # Hips are on floor, chest lifted. This creates an obtuse angle (>90)
        "back_extension": 135.0,
        # Elbows bent, tucked near ribs
        "left_elbow_bend": 110.0,
    }
}



def compute_pose_angles(landmarks, pose_map):
    """
    landmarks: list-like of landmark objects with .x, .y (normalized)
    pose_map: mapping like TARGET_POSES['warrior_ii']
    returns dict of computed angles
    """
    angles = {}
    for key, (ia, ib, ic) in pose_map.items():
        # Wrap in try/except in case a specific landmark isn't detected
        try:
            pa = (landmarks[ia].x, landmarks[ia].y)
            pb = (landmarks[ib].x, landmarks[ib].y)
            pc = (landmarks[ic].x, landmarks[ic].y)
            angles[key] = angle_between_points(pa, pb, pc)
        except IndexError:
             # Handle cases where a limb might be off-screen
             angles[key] = 0.0

    return angles


def score_against_target(computed, target):
    """
    computed, target: dict of joint->angle
    returns overall accuracy 0-100 and per-joint diffs
    """
    diffs = {}
    total_score = 0.0
    count = 0
    # Defines how many degrees off is considered "0 score".
    # Stricter poses might need lower tolerance.
    TOLERANCE_DEG = 35.0

    for k, tgt_angle in target.items():
        if k in computed:
            observed_angle = computed[k]
            d = abs(observed_angle - tgt_angle)

            # Simple linear scoring:
            # If diff is 0, score is 100.
            # If diff is >= TOLERANCE_DEG, score is 0.
            score = max(0.0, (1.0 - (d / TOLERANCE_DEG))) * 100.0

            diffs[k] = {
                "target": round(tgt_angle, 1),
                "observed": round(observed_angle, 1),
                "diff": round(d, 1),
                "score": round(score, 1)
            }
            total_score += score
            count += 1

    overall = (total_score / count) if count else 0.0
    return overall, diffs

# --- Django Views ---

# Page that serves the live pose UI
def pose_live_page(request):
    # Pass available poses to template if needed for a dropdown selector
    context = {"available_poses": list(TARGET_POSES.keys())}
    return render(request, "yoga_app/pose_live.html", context)


# Endpoint to accept base64 image frames and return accuracy
@csrf_exempt
def analyze_frame(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    if mp_pose is None:
        return JsonResponse({"error": "MediaPipe not available on server."}, status=500)

    try:
        data = json.loads(request.body.decode("utf-8"))
        b64 = data.get("image")
        # Default to warrior_ii if pose not specified
        selected_pose = data.get("pose", "warrior_ii")

        if not b64:
            return HttpResponseBadRequest("No image provided.")

        # data URL like "data:image/png;base64,......"
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        # MediaPipe expects RGB. Using a slightly higher model complexity for better accuracy.
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1) as pose:
            results = pose.process(img_np)
            if not results.pose_landmarks:
                return JsonResponse({"error": "no_landmarks"})

            landmarks = results.pose_landmarks.landmark

            # 1. Get configuration for selected pose
            pose_map = TARGET_POSES.get(selected_pose)
            angle_targets = POSE_ANGLE_TARGETS.get(selected_pose)

            if not pose_map or not angle_targets:
                 return JsonResponse({
                    "error": f"Pose '{selected_pose}' not recognized.",
                    "available_poses": list(TARGET_POSES.keys())
                }, status=400)

            # 2. Compute angles specific to this pose
            computed = compute_pose_angles(landmarks, pose_map)

            # 3. Score against targets
            overall, per_joint = score_against_target(computed, angle_targets)

            # return nice numbers
            return JsonResponse({
                "pose": selected_pose,
                "overall_score": round(overall, 1),
                "per_joint": per_joint,
            })
    except json.JSONDecodeError:
         return HttpResponseBadRequest("Invalid JSON")
    except Exception as e:
        print(f"Analysis Error: {e}") # Log error for server side debugging
        return JsonResponse({"error": "exception", "detail": str(e)}, status=500)