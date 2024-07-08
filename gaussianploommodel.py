import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 가우시안 플룸 모델 함수
def gaussian_plume(Q, u, H, x, y, z, stability_class='D'):
    stability_classes = {
        'A': (0.22, 0.20),
        'B': (0.16, 0.12),
        'C': (0.11, 0.08),
        'D': (0.08, 0.06),
        'E': (0.06, 0.03),
        'F': (0.04, 0.016)
    }
    sigma_y, sigma_z = stability_classes[stability_class]
    sigma_y *= x**0.894
    sigma_z *= x**0.894
    
    C = (Q / (2 * np.pi * sigma_y * sigma_z * u)) * \
        np.exp(-y**2 / (2 * sigma_y**2)) * \
        (np.exp(-(z - H)**2 / (2 * sigma_z**2)) + np.exp(-(z + H)**2 / (2 * sigma_z**2)))
    
    return C

# 기상청 기상 정보 API URL
api_key = "ownkey" 
tm = "201909060900"  # 2019년 9월 6일 오전 9시
stn = "133"  # 대전광역시 유성구
help_flag = "1"

weather_url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php?tm={tm}&stn={stn}&help={help_flag}&authKey={api_key}"


response_weather = requests.get(weather_url)


print("Response Status Code:", response_weather.status_code)


print("Response Content:", response_weather.text)


lines = response_weather.text.split('\n')


start_index = None
end_index = None
for i, line in enumerate(lines):
    if line.startswith("# YYMMDDHHMI"):
        start_index = i + 1
    elif line.startswith("#7777END"):
        end_index = i


data_lines = lines[start_index:end_index]


data = []
for line in data_lines:

    row = line.split()
   
    if len(row) == 46: 
        data.append(row)

columns = [
    "TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM", "PA", "PS", "PT", "PR", 
    "TA", "TD", "HM", "PV", "RN", "RN_DAY", "RN_JUN", "RN_INT", "SD_HR3", "SD_DAY",
    "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID", "CH_MIN", "CT", "CT_TOP", 
    "CT_MID", "CT_LOW", "VS", "SS", "SI", "ST_GD", "TS", "TE_005", "TE_01", 
    "TE_02", "TE_03", "ST_SEA", "WH", "BF", "IR", "IX"
]

df_weather = pd.DataFrame(data, columns=columns)

df_weather_filtered = df_weather[['TM', 'WS', 'WD']]


df_weather_filtered = df_weather_filtered[df_weather_filtered['TM'].apply(lambda x: x.isdigit())]


df_weather_filtered['time'] = pd.to_datetime(df_weather_filtered['TM'], format='%Y%m%d%H%M')
df_weather_filtered['WS'] = df_weather_filtered['WS'].astype(float)
df_weather_filtered['WD'] = df_weather_filtered['WD'].astype(float)


average_ws = df_weather_filtered['WS'].mean()
average_wd = df_weather_filtered['WD'].mean()

print(f"Average Wind Speed: {average_ws} m/s")
print(f"Average Wind Direction: {average_wd} degrees")


Q = 1000  
u = average_ws 
H = 20
x = np.linspace(0, 5000, 500)  
y = np.linspace(-1000, 1000, 200)  
X, Y = np.meshgrid(x, y)


Z = gaussian_plume(Q, u, H, X, Y, z=0, stability_class='D')


plt.figure(figsize=(12, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='jet')
plt.colorbar(contour, label='Concentration (µg/m³)')
plt.xlabel('Distance downwind (m)')
plt.ylabel('Lateral distance (m)')
plt.title('Gaussian Plume Model for Daejeon Hanwha Explosion Incident')
plt.show()
