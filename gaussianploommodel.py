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
api_key = "W5YeIK2aRAWWHiCtmsQFBQ"  # 실제 발급받은 API 키로 대체
tm = "201909060900"  # 2019년 9월 6일 오전 9시
stn = "133"  # 대전광역시 유성구 지점번호 (예시)
help_flag = "1"

weather_url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php?tm={tm}&stn={stn}&help={help_flag}&authKey={api_key}"

# 기상 데이터 수집
response_weather = requests.get(weather_url)

# 응답 상태 코드 출력
print("Response Status Code:", response_weather.status_code)

# 응답 내용 출력 (텍스트 형식)
print("Response Content:", response_weather.text)

# 텍스트 데이터를 행 단위로 분리
lines = response_weather.text.split('\n')

# 데이터 시작과 끝을 찾아 추출
start_index = None
end_index = None
for i, line in enumerate(lines):
    if line.startswith("# YYMMDDHHMI"):
        start_index = i + 1
    elif line.startswith("#7777END"):
        end_index = i

# 데이터 추출
data_lines = lines[start_index:end_index]

# 각 행을 공백으로 분리하여 데이터프레임으로 변환
data = []
for line in data_lines:
    # 공백을 기준으로 데이터 분리
    row = line.split()
    # 유효한 데이터 행만 추가 (행의 길이 검사)
    if len(row) == 46:  # 열의 개수는 데이터에 따라 다를 수 있으므로 확인 필요
        data.append(row)

# 데이터프레임 생성
columns = [
    "TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM", "PA", "PS", "PT", "PR", 
    "TA", "TD", "HM", "PV", "RN", "RN_DAY", "RN_JUN", "RN_INT", "SD_HR3", "SD_DAY",
    "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID", "CH_MIN", "CT", "CT_TOP", 
    "CT_MID", "CT_LOW", "VS", "SS", "SI", "ST_GD", "TS", "TE_005", "TE_01", 
    "TE_02", "TE_03", "ST_SEA", "WH", "BF", "IR", "IX"
]

df_weather = pd.DataFrame(data, columns=columns)

# 필요한 기상 데이터 필터링 (바람 속도 및 방향)
df_weather_filtered = df_weather[['TM', 'WS', 'WD']]

# 유효한 시간 형식의 데이터만 필터링
df_weather_filtered = df_weather_filtered[df_weather_filtered['TM'].apply(lambda x: x.isdigit())]

# 시간대별 데이터 변환
df_weather_filtered['time'] = pd.to_datetime(df_weather_filtered['TM'], format='%Y%m%d%H%M')
df_weather_filtered['WS'] = df_weather_filtered['WS'].astype(float)
df_weather_filtered['WD'] = df_weather_filtered['WD'].astype(float)

# 평균 바람 속도와 방향 계산
average_ws = df_weather_filtered['WS'].mean()
average_wd = df_weather_filtered['WD'].mean()

print(f"Average Wind Speed: {average_ws} m/s")
print(f"Average Wind Direction: {average_wd} degrees")

# 모델 파라미터 설정 (대전 한화 공장 폭발 사고 시)
Q = 1000  # 방출율 (g/s) 예시 값
u = average_ws  # 평균 바람 속도 (m/s)
H = 20  # 오염원 높이 (m)
x = np.linspace(0, 5000, 500)  # 바람 방향으로의 거리 (m)
y = np.linspace(-1000, 1000, 200)  # y축 위치 (m)
X, Y = np.meshgrid(x, y)

# 오염 농도 계산
Z = gaussian_plume(Q, u, H, X, Y, z=0, stability_class='D')

# 결과 시각화
plt.figure(figsize=(12, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='jet')
plt.colorbar(contour, label='Concentration (µg/m³)')
plt.xlabel('Distance downwind (m)')
plt.ylabel('Lateral distance (m)')
plt.title('Gaussian Plume Model for Daejeon Hanwha Explosion Incident')
plt.show()
