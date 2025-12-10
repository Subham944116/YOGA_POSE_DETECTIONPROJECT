from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('detect/', views.detect_pose, name='detect_pose'),
    path("info/",views.info, name="info"),
    path("pose/live/", views.pose_live_page, name="pose_live_page"),
    path("pose/analyze-frame/", views.analyze_frame, name="pose_analyze_frame"),

]
