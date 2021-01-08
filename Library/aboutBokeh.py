from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show
from bokeh.io import curdoc
from bokeh.embed import components
import numpy as np

## bokeh
# 시각화를 도와주는 파이썬 라이브러리
# 반응형, 다양한 디자인

## 기본적인 사용 방법
x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

# HTML 파일을 생성해서 시각화 확인
output_file("bokeh.html")

# figure 생성
p = figure(title = "simple line example", x_axis_label = 'x', y_axis_label = 'y')

# figure에 그릴 그래프 생성
p.line(x, y, legend = "Temp.", line_width = 2)

show(p)

## Embed
# 본인의 홈페이지에 embed 하기
# 자바스크립트 형태로 그래프 출력

# 예.
plot = figure()
plot.circle([1, 2], [3, 4])

# script, div를 홈페이지에 복사 붙여넣기 하면 가능
script, div = components(plot)
print(script, div)

## numpy와 bokeh를 함께 사용하기
output_file("bokeh.html")

N = 4000
x = np.random.random(size = N) * 100
y = np.random.random(size = N) * 100
radii = np.random.random(size = N) * 1.5
# 그라디언트 색상 적용
colors = ["#%02x%02x%02x" % (r, g, 150) for r, g in zip(np.floor(50 + 2 * x).astype(int), np.floor(30 + 2 * y).astype(int))]

p = figure()
p.circle(x, y, radius=radii, fill_color=colors, fill_alpha=0.6, line_color=None)
show(p)

# ---------------------------------------------------------------#

# Function 형식으로 만들기
def modify_doc(doc):
    """Add a plotted function to the document.

    Arguments:
        doc: A bokeh document to which elements can be added.
    """
    x_values = range(10)
    y_values = [x ** 2 for x in x_values]
    data_source = ColumnDataSource(data=dict(x=x_values, y=y_values))
    plot = figure(title="f(x) = x^2",
                  tools="crosshair,pan,reset,save,wheel_zoom",)
    plot.line('x', 'y', source=data_source, line_width=3, line_alpha=0.6)
    doc.add_root(plot)
    doc.title = "Hello World"

def main():
    modify_doc(curdoc())
    
main()
# bokeh serve --show Library/aboutBokeh.py로 실행

