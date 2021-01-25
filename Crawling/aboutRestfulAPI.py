from flask import Flask
from flask_restful import Resource, reqparse, Api

# Restful 설정
app = Flask(__name__)
api = Api(app)

# Route 설정
# Request, Response 설정
@app.route('/chatbot/<chat>')
def chatbot(chat):
    try:
        return {'status': 'success', 'request': chat}
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    app.run(debug=True)

# http://127.0.0.1:5000/chatbot/로 접속하여 뒤에 test라고 입력할 경우
# {
#   "request": "test",
#   "status": "success"
# }
# 이 출력


#####################################################

class Plus(Resource):
    def get(self):
        try:
            parser = reqparse.RequestParser()
            
            # required => 필수, type => 입력 타입 (숫자만)
            parser.add_argument('x', required=True, type=int, help='x cannot be blank')
            parser.add_argument('y', required=True, type=int, help='y cannot be blank')
            
            args = parser.parse_args()
            
            result = args['x'] + args['y']
            
            return {'result': result}
        except Exception as e:
            return {'error': str(e)}

app = Flask('First')
api = Api(app)
api.add_resource(Plus, '/plus')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

# 실행 후, localhost:8000/plus?x=2&y=3 라고 입력 시
# x=2&y=3 => parse 값 전달 (argument)
# {
#     "result": 5
# }
# 의 결과가 출력