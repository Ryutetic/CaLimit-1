from flask import Flask, render_template, request, jsonify
import sympy as sp
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import re
from fractions import Fraction
from functools import lru_cache

app = Flask(__name__)

# ========== MODULE 1: COLOR & CONFIG ==========
COLORS = {
    'bg': '#FFF5E1', 'accent': '#D7A58F', 'secondary': '#A67C52',
    'text': '#5C3A21', 'plot_bg': '#FDF6F0', 'button': '#8B5A2B',
    'button_hover': '#6D4424', 'card': '#FAEBD7', 'error': '#FF6B6B',
    'success': '#4ECDC4', 'warning': '#FFE66D', 'left': '#4ECDC4',
    'right': '#FF6B6B', 'both': '#45B7D1', 'theory': '#6A89CC',
    'trig': '#FF9FF3', 'infinity': '#706FD3', 'threed': '#FF9F1C'
}
FONT = "'Nunito', sans-serif"

# ========== MODULE 2: PARSING & VALIDATION ==========
class ExpressionParser:
    ALLOWED_SYMBOLS = {'x', 'y', 'pi', 'e', 'oo'}
    
    @staticmethod
    @lru_cache(maxsize=100)
    def parse(expr):
        if not expr or not expr.strip():
            return {"error": "❌ Ekspresi tidak boleh kosong"}
        
        try:
            expr = ExpressionParser._sanitize(expr)
            expr = ExpressionParser._preprocess(expr)
            expr = ExpressionParser._validate_symbols(expr)
            return {"success": expr}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def _sanitize(expr):
        dangerous = [r'__.*__', r'import\s+\w+', r'exec\s*\(', r'eval\s*\(',
                    r'open\s*\(', r'file\s*\(', r'os\.', r'sys\.', r'subprocess\.']
        
        for pattern in dangerous:
            expr = re.sub(pattern, '', expr, flags=re.IGNORECASE)
        
        if len(expr) > 500:
            raise ValueError("❌ Ekspresi terlalu panjang (max 500 karakter)")
        
        return expr.strip()
    
    @staticmethod
    def _preprocess(expr):
        original = expr
        try:
            expr = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr)
            expr = re.sub(r'([a-zA-Z)])(\d)', r'\1*\2', expr)
            expr = re.sub(r'([a-zA-Z)])([a-zA-Z(])', r'\1*\2', expr)
            
            trig_map = {'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'cot': 'cot',
                       'sec': 'sec', 'csc': 'csc', 'arcsin': 'asin', 'arccos': 'acos', 
                       'arctan': 'atan', 'ln': 'log', '√': 'sqrt'}
            
            for natural, sympy_func in trig_map.items():
                expr = re.sub(rf'{natural}\s*\(', f'{sympy_func}(', expr, flags=re.IGNORECASE)
            
            expr = expr.replace('²', '**2').replace('³', '**3')
            expr = expr.replace('π', 'pi').replace('∞', 'oo').replace('infinity', 'oo')
            expr = expr.replace('e^', 'exp').replace('^', '**')
            expr = re.sub(r'\s+', '', expr)
            
            return expr
        except Exception as e:
            raise ValueError(f"❌ Gagal parse ekspresi: {original}")
    
    @staticmethod
    def _validate_symbols(expr):
        try:
            test_expr = sp.sympify(expr)
            free_symbols = {str(s) for s in test_expr.free_symbols}
            allowed = ExpressionParser.ALLOWED_SYMBOLS
            invalid_symbols = free_symbols - allowed
            
            if invalid_symbols:
                raise ValueError(f"❌ Simbol tidak diizinkan: {invalid_symbols}")
                
            return expr
        except sp.SympifyError as e:
            raise ValueError(f"❌ Ekspresi tidak valid")

# ========== MODULE 3: LIMIT CALCULATOR ==========
class LimitCalculator:
    @staticmethod
    @lru_cache(maxsize=100)
    def calculate(expr_str, x_val, direction='both'):
        x = sp.symbols('x')
        
        try:
            parse_result = ExpressionParser.parse(expr_str)
            if "error" in parse_result:
                return {"error": parse_result["error"]}
                
            parsed_expr = parse_result["success"]
            expr = sp.sympify(parsed_expr)
            
            if x_val == 'oo':
                x_val_sym = sp.oo
            elif x_val == '-oo':
                x_val_sym = -sp.oo
            else:
                x_val_sym = x_val
            
            if direction == 'both':
                result = sp.limit(expr, x, x_val_sym)
            elif direction == 'left':
                result = sp.limit(expr, x, x_val_sym, dir='-')
            else:
                result = sp.limit(expr, x, x_val_sym, dir='+')
            
            return {"success": LimitCalculator._format_result(result)}
            
        except Exception as e:
            return {"error": f"Error: {str(e)}"}
    
    @staticmethod
    def _format_result(value):
        if value == sp.oo:
            return '∞'
        elif value == -sp.oo:
            return '-∞'
        elif value in [sp.zoo, sp.nan]:
            return 'Tidak terdefinisi'
        
        try:
            float_val = float(value.evalf())
            if float_val == int(float_val):
                return str(int(float_val))
            
            frac = Fraction(float_val).limit_denominator(100)
            if frac.denominator != 1 and abs(frac.numerator/frac.denominator - float_val) < 1e-10:
                return f"{frac.numerator}/{frac.denominator}"
            
            return f"{float_val:.6f}".rstrip('0').rstrip('.')
        except:
            return str(value)

# ========== MODULE 4: PLOTTING ENGINE ==========
class PlottingEngine:
    @staticmethod
    def create_2d_plot(expr_str1, expr_str2, x_val, direction):
        if x_val in ['oo', '-oo', '∞', '-∞']:
            return PlottingEngine._no_plot_message("Visualisasi tidak tersedia untuk limit di tak hingga")
        
        try:
            x_val_float = float(x_val)
            x = sp.symbols('x')
            
            left_range = max(x_val_float - 3, x_val_float - 0.1) 
            right_range = min(x_val_float + 3, x_val_float + 0.1)
            
            X = np.linspace(left_range, right_range, 500)
            fig = go.Figure()
            
            colors = [COLORS['accent'], COLORS['secondary']]
            expressions = [(expr_str1, 'f(x)'), (expr_str2, 'g(x)')]
            
            for i, (expr_str, label) in enumerate(expressions):
                if expr_str and expr_str.strip():
                    try:
                        f = PlottingEngine._get_cached_function(expr_str, [x])
                        Y = PlottingEngine._evaluate_function_vectorized(f, X)
                        
                        display_expr = ExpressionFormatter.format_display(expr_str)
                        fig.add_trace(go.Scatter(
                            x=X, y=Y, mode='lines', 
                            line=dict(color=colors[i], width=3),
                            name=f'{label} = {display_expr}'
                        ))
                    except:
                        continue
            
            PlottingEngine._add_limit_visualization(fig, x_val_float, direction)
            
            fig.update_layout(
                plot_bgcolor=COLORS['plot_bg'],
                paper_bgcolor=COLORS['plot_bg'],
                font=dict(color=COLORS['text'], family=FONT),
                title="📈 Visualisasi 2D - Limit & Pendekatan Fungsi",
                title_x=0.5, height=500, showlegend=True
            )
            
            return pio.to_html(fig, full_html=False)
            
        except Exception as e:
            return PlottingEngine._error_plot_message(f"Error plot 2D: {str(e)}")
    
    @staticmethod
    def create_3d_plot(expr_str):
        try:
            x, y = sp.symbols('x y')
            
            if not PlottingEngine._is_3d_function(expr_str):
                return PlottingEngine._no_plot_message("Fungsi tidak mengandung variabel y untuk visualisasi 3D")
            
            f = PlottingEngine._get_cached_function(expr_str, [x, y])
            
            X = np.linspace(-3, 3, 50)
            Y = np.linspace(-3, 3, 50)
            X, Y = np.meshgrid(X, Y)
            
            try:
                Z = f(X, Y)
                Z = np.where(np.isfinite(Z), Z, np.nan)
            except:
                Z = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        try:
                            Z[i,j] = f(X[i,j], Y[i,j])
                        except:
                            Z[i,j] = np.nan
            
            fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8)])
            
            fig.update_layout(
                title='🎨 Visualisasi 3D (Drag untuk rotate)',
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='f(x,y)'),
                font=dict(family=FONT, color=COLORS['text']), height=500
            )
            
            return pio.to_html(fig, full_html=False)
            
        except Exception as e:
            return PlottingEngine._error_plot_message(f"Error plot 3D: {str(e)}")
    
    @staticmethod
    @lru_cache(maxsize=50)
    def _get_cached_function(expr_str, variables):
        parse_result = ExpressionParser.parse(expr_str)
        if "error" in parse_result:
            raise ValueError(parse_result["error"])
        parsed_expr = parse_result["success"]
        expr = sp.sympify(parsed_expr)
        return sp.lambdify(variables, expr, modules=['numpy'])
    
    @staticmethod
    def _evaluate_function_vectorized(f, X):
        try:
            Y = f(X)
            return np.where(np.isfinite(Y), Y, np.nan)
        except:
            Y = []
            for v in X:
                try:
                    Y.append(f(v) if np.isfinite(f(v)) else np.nan)
                except:
                    Y.append(np.nan)
            return np.array(Y)
    
    @staticmethod
    def _is_3d_function(expr_str):
        try:
            parse_result = ExpressionParser.parse(expr_str)
            if "error" in parse_result:
                return False
            parsed_expr = parse_result["success"]
            expr = sp.sympify(parsed_expr)
            return 'y' in str(expr)
        except:
            return False
    
    @staticmethod
    def _add_limit_visualization(fig, x_val, direction):
        if direction in ['left', 'both']:
            fig.add_vline(x=x_val - 0.001, line_dash="dot", line_color=COLORS['left'], opacity=0.5)
        if direction in ['right', 'both']:
            fig.add_vline(x=x_val + 0.001, line_dash="dot", line_color=COLORS['right'], opacity=0.5)
        fig.add_vline(x=x_val, line_dash="dash", line_color="gray", opacity=0.7)
    
    @staticmethod
    def _no_plot_message(message):
        return f"<div style='padding:20px; background:{COLORS['warning']}20; border-radius:10px; text-align:center;'>{message}</div>"
    
    @staticmethod
    def _error_plot_message(message):
        return f"<div style='padding:20px; background:{COLORS['error']}20; border-radius:10px; text-align:center;'>{message}</div>"

# ========== MODULE 5: CONTINUITY ANALYZER ==========
class ContinuityAnalyzer:
    @staticmethod
    def analyze(expr_str, x_val):
        x = sp.symbols('x')
        analysis = {'steps': [], 'continuous': False, 'discontinuity_type': None}
        
        try:
            parse_result = ExpressionParser.parse(expr_str)
            if "error" in parse_result:
                analysis['steps'].append(f"❌ {parse_result['error']}")
                return analysis
                
            parsed_expr = parse_result["success"]
            expr = sp.sympify(parsed_expr)
            
            analysis['steps'].append(f"🔍 <b>Analisis Kontinuitas di x = {x_val}:</b>")
            
            defined, func_value = ContinuityAnalyzer._check_definition(expr, x, x_val)
            analysis['defined'] = defined
            analysis['func_value'] = func_value
            
            limit_result_left = LimitCalculator.calculate(expr_str, x_val, 'left')
            limit_result_right = LimitCalculator.calculate(expr_str, x_val, 'right')
            limit_result_both = LimitCalculator.calculate(expr_str, x_val, 'both')
            
            analysis.update({
                'limit_left': limit_result_left.get('success', limit_result_left.get('error', 'Error')),
                'limit_right': limit_result_right.get('success', limit_result_right.get('error', 'Error')),
                'limit_both': limit_result_both.get('success', limit_result_both.get('error', 'Error'))
            })
            
            analysis['steps'].extend([
                f"📊 Limit kiri: {analysis['limit_left']}",
                f"📊 Limit kanan: {analysis['limit_right']}",
                f"📊 Limit kedua arah: {analysis['limit_both']}"
            ])
            
            continuity_result = ContinuityAnalyzer._determine_continuity(analysis, x_val)
            analysis.update(continuity_result)
            
            analysis['steps'].append(f"<b>Kesimpulan:</b> {analysis['discontinuity_type']}")
            
        except Exception as e:
            analysis['steps'].append(f"❌ Error dalam analisis: {str(e)}")
        
        return analysis
    
    @staticmethod
    def _check_definition(expr, x, x_val):
        if x_val in ['oo', '-oo']:
            return False, "Tidak terdefinisi"
        
        try:
            func_val = expr.subs(x, x_val)
            if func_val.is_infinite or func_val.is_complex:
                return False, "Tidak terdefinisi"
            return True, LimitCalculator._format_result(func_val)
        except:
            return False, "Tidak terdefinisi"
    
    @staticmethod
    def _determine_continuity(analysis, x_val):
        if x_val in ['oo', '-oo']:
            return {'continuous': False, 'discontinuity_type': "Titik tak hingga"}
        
        if not analysis['defined']:
            if (analysis['limit_left'] == analysis['limit_right'] and 
                analysis['limit_left'] not in ['Error:', 'Tidak terdefinisi', '∞', '-∞']):
                return {'continuous': False, 'discontinuity_type': "Diskontinuitas yang dapat dihapus"}
            else:
                return {'continuous': False, 'discontinuity_type': "Diskontinuitas esensial"}
        
        limit_exists = (analysis['limit_left'] == analysis['limit_right'] == analysis['limit_both'] and
                      analysis['limit_both'] not in ['Error:', 'Tidak terdefinisi', '∞', '-∞'])
        
        limit_equals_value = (analysis['limit_both'] == analysis['func_value'])
        
        if limit_exists and limit_equals_value:
            return {'continuous': True, 'discontinuity_type': "Kontinu"}
        else:
            if not limit_exists:
                return {'continuous': False, 'discontinuity_type': "Diskontinuitas lompatan"}
            else:
                return {'continuous': False, 'discontinuity_type': "Diskontinuitas yang dapat dihapus"}

# ========== MODULE 6: UI COMPONENTS ==========
class ExpressionFormatter:
    @staticmethod
    def format_display(expr_str):
        if not expr_str:
            return ""
        
        expr_str = str(expr_str)
        replacements = [
            (r'\*\*', '^'), (r'sqrt\(([^)]+)\)', r'√(\1)'),
            (r'pi', 'π'), (r'oo', '∞'), (r'exp\(([^)]+)\)', r'e^(\1)'),
            (r'asin', 'arcsin'), (r'acos', 'arccos'), (r'atan', 'arctan'),
            (r'\^2', '²'), (r'\^3', '³'), (r'(\d)\*', r'\1')
        ]
        
        for pattern, replacement in replacements:
            expr_str = re.sub(pattern, replacement, expr_str)
        
        return expr_str

class InputValidator:
    @staticmethod
    def parse_x_value(x_str):
        if not x_str:
            return None, "Nilai x tidak boleh kosong"
        
        x_str = x_str.strip()
        
        if x_str.lower() in ['inf', 'infinity', '∞']:
            return 'oo', None
        elif x_str.lower() in ['-inf', '-infinity', '-∞']:
            return '-oo', None
        
        if x_str.endswith('+') or x_str.endswith('-'):
            try:
                base_val = float(x_str[:-1])
                direction = 'right' if x_str.endswith('+') else 'left'
                return base_val, direction
            except:
                return None, "Format nilai x tidak valid"
        
        try:
            return float(x_str), None
        except:
            return None, "Nilai x harus berupa angka"

class ExampleManager:
    EXAMPLES = [
        {"name": "🔺 Limit Trigonometri", "expr": "sin(x)/x", "xval": "0"},
        {"name": "∞ Limit Tak Hingga", "expr": "(2x^2 + 3)/(x^2 - 1)", "xval": "infinity"},
        {"name": "√ Limit Akar", "expr": "sqrt(x-2)", "xval": "2+"},
        {"name": "📈 Limit Eksponensial", "expr": "(1 + 1/x)^x", "xval": "infinity"},
        {"name": "🎭 Limit Piecewise", "expr": "(x^2 - 1)/(x - 1)", "xval": "1"},
        {"name": "🧊 Fungsi 3D", "expr": "x^2 + y^2", "xval": "0"}
    ]

# ========== FLASK ROUTES ==========
@app.route('/')
def index():
    examples = ExampleManager.EXAMPLES
    return render_template('index.html', colors=COLORS, font=FONT, examples=examples)

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    expr1 = data.get('expr1', '')
    expr2 = data.get('expr2', '')
    xval_str = data.get('xval', '')
    direction = data.get('direction', 'both')
    
    # Validate input
    xval, auto_direction = InputValidator.parse_x_value(xval_str)
    if xval is None:
        return jsonify({"error": auto_direction})
    
    if auto_direction:
        direction = auto_direction
    
    # Validate expression
    parse_result = ExpressionParser.parse(expr1)
    if "error" in parse_result:
        return jsonify({"error": parse_result["error"]})
    
    # Calculate results
    results = {}
    
    # Function 1
    limit_result1 = LimitCalculator.calculate(expr1, xval, direction)
    continuity1 = ContinuityAnalyzer.analyze(expr1, xval)
    
    results['function1'] = {
        'expression': expr1,
        'display_expression': ExpressionFormatter.format_display(expr1),
        'limit': limit_result1.get('success') if 'success' in limit_result1 else limit_result1.get('error'),
        'continuity': continuity1
    }
    
    # Function 2 (if provided)
    if expr2 and expr2.strip():
        parse_result2 = ExpressionParser.parse(expr2)
        if "error" not in parse_result2:
            limit_result2 = LimitCalculator.calculate(expr2, xval, direction)
            continuity2 = ContinuityAnalyzer.analyze(expr2, xval)
            
            results['function2'] = {
                'expression': expr2,
                'display_expression': ExpressionFormatter.format_display(expr2),
                'limit': limit_result2.get('success') if 'success' in limit_result2 else limit_result2.get('error'),
                'continuity': continuity2
            }
    
    # Generate plots
    plot_2d = PlottingEngine.create_2d_plot(expr1, expr2, xval, direction)
    plot_3d = PlottingEngine.create_3d_plot(expr1)
    
    results['plots'] = {
        'plot_2d': plot_2d,
        'plot_3d': plot_3d
    }
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
