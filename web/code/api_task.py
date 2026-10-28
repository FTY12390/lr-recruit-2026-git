from flask import Flask, request, jsonify

# 1. 初始化 Flask 应用
app = Flask(__name__)

# 2. 定义路由与请求方法
@app.route('/calculate', methods=['POST'])
def calculate():
    # 3. 从表单中提取数据
    operation = request.form.get('Operation')
    data_a_str = request.form.get('Data_A')
    data_b_str = request.form.get('Data_B')

    # 4. 参数基础校验（防止漏传）
    if not all([operation, data_a_str, data_b_str]):
        return jsonify({"error": "缺少参数，请提供 Operation, Data_A, Data_B"}), 400

    # 5. 数据类型转换与异常捕获
    try:
        data_a = float(data_a_str)
        data_b = float(data_b_str)
    except ValueError:
        return jsonify({"error": "Data_A 和 Data_B 必须是数字"}), 400

    # 6. 核心业务逻辑处理
    result = None
    try:
        if operation == 'sum':
            result = data_a + data_b
        elif operation == 'subtract':
            result = data_a - data_b
        elif operation == 'multiply':
            result = data_a * data_b
        elif operation == 'divide':
            if data_b == 0:
                return jsonify({"error": "除数不能为 0"}), 400
            result = data_a / data_b
        else:
            return jsonify({"error": f"不支持的操作: {operation}，请使用 sum, subtract, multiply, divide"}), 400
    except Exception as e:
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500

    # 7. 构造响应并返回
    return jsonify({"Result": result}), 200

# 8. 启动服务
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

