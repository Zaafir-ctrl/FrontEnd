from django.urls import path

from . import views

app_name = "FootballPredictions"

urlpatterns = [

    # path('',views.Team_Based_Prediction),
    path('',views.Football_Landing),
    path('maps/',views.maps),
    path('feedback/',views.feedback),
    path('what_is_deepshot/',views.what_is_deepshot),
    path('maps/netherlands/',views.netherlands),
    path('maps/argentina/',views.argentina),
    path('maps/cameroon/',views.cameroon),
    path('maps/croatia/',views.croatia),
    path('maps/japan/',views.japan),
    path('maps/man_u/',views.man_u),
    path('maps/morocco/',views.morocco),
    path('maps/tot/',views.tot),
    path('maps/wales/',views.wales),
    # path('maps/netherlands/',views.netherlands),
    # path('maps/netherlands/',views.netherlands),
    
    path('maps/portugal/',views.portugal),
    path('team_based/barcelona/',views.Team_Based_barcelona)

]
