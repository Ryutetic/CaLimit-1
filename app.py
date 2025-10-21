from flask import Flask, render_template, request, jsonify
import sympy as sp
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import re
from fractions import Fraction
from functools import lru_cache
import math
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

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
            # Tambahkan operator * yang hilang
            expr = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr)
            expr = re.sub(r'([a-zA-Z)])(\d)', r'\1*\2', expr)
            expr = re.sub(r'([a-zA-Z)])([a-zA-Z(])', r'\1*\2', expr)
            
            # Konversi fungsi trigonometri
            trig_map = {'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'cot': 'cot',
                       'sec': 'sec', 'csc': 'csc', 'arcsin': 'asin', 'arccos': 'acos', 
                       'arctan': 'atan', 'ln': 'log', '√': 'sqrt', 'log': 'log'}
            
            for natural, sympy_func in trig_map.items():
                expr = re.sub(rf'{natural}\s*\(', f'{sympy_func}(', expr, flags=re.IGNORECASE)
            
            # Konversi notasi
            expr = expr.replace('²', '**2').replace('³', '**3')
            expr = expr.replace('π', 'pi').replace('∞', 'oo').replace('infinity', 'oo')
            expr = expr.replace('e^', 'exp').replace('^', '**')
            expr = re.sub(r'\s+', '', expr)
            
            return expr
        except Exception as e:
            raise ValueError(f"❌ Gagal parse ekspresi: {original}. Error: {str(e)}")
    
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
            raise ValueError(f"❌ Ekspresi tidak valid: {str(e)}")

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
            
            # Konversi nilai x
            if x_val == 'oo':
                x_val_sym = sp.oo
            elif x_val == '-oo':
                x_val_sym = -sp.oo
            else:
                try:
                    x_val_sym = float(x_val)
                except:
                    x_val_sym = sp.sympify(x_val)
            
            # Hitung limit
            if direction == 'both':
                result = sp.limit(expr, x, x_val_sym)
            elif direction == 'left':
                result = sp.limit(expr, x, x_val_sym, dir='-')
            else:
                result = sp.limit(expr, x, x_val_sym, dir='+')
            
            return {"success": LimitCalculator._format_result(result)}
            
        except Exception as e:
            return {"error": f"Error menghitung limit: {str(e)}"}
    
    @staticmethod
    def _format_result(value):
        if value == sp.oo:
            return '∞'
        elif value == -sp.oo:
            return '-∞'
        elif value in [sp.zoo, sp.nan]:
            return 'Tidak terdefinisi'
        elif value.is_real:
            try:
                float_val = float(value)
                if abs(float_val) > 1e10:
                    return '∞' if float_val > 0 else '-∞'
                
                # Coba format sebagai pecahan dengan presisi lebih tinggi
                frac = Fraction(float_val).limit_denominator(10000)
                if abs(frac.numerator/frac.denominator - float_val) < 1e-12:
                    if frac.denominator == 1:
                        return str(frac.numerator)
                    else:
                        return f"{frac.numerator}/{frac.denominator}"
                
                # Format desimal dengan presisi
                formatted = f"{float_val:.8f}"
                # Hapus trailing zeros
                if '.' in formatted:
                    formatted = formatted.rstrip('0').rstrip('.')
                return formatted
            except:
                return str(value)
        else:
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
            
            # RANGE OPTIMAL untuk berbagai kasus
            if x_val_float == 0:
                left_range, right_range = -3, 3
            elif abs(x_val_float) < 1:
                left_range, right_range = -2, 2
            else:
                range_size = max(2, abs(x_val_float) * 0.6)
                left_range = x_val_float - range_size
                right_range = x_val_float + range_size
            
            # BUAT TITIK DENGAN STRATEGI PINTAR
            X_main = np.linspace(left_range, right_range, 400)
            
            # Titik padat di sekitar limit point
            epsilon = 0.2
            X_near = np.linspace(x_val_float - epsilon, x_val_float + epsilon, 200)
            
            # Gabungkan dan urutkan
            X = np.sort(np.concatenate([X_main, X_near]))
            
            # Hindari titik tepat di singularitas
            X = X[abs(X - x_val_float) > 1e-8]
            
            fig = go.Figure()
            
            colors = ['#FF6B6B', '#4ECDC4']  # Warna fixed
            expressions = [(expr_str1, 'f(x)'), (expr_str2, 'g(x)')]
            
            valid_plots = 0
            
            for i, (expr_str, label) in enumerate(expressions):
                if expr_str and expr_str.strip():
                    try:
                        parse_result = ExpressionParser.parse(expr_str)
                        if "error" in parse_result:
                            continue
                            
                        parsed_expr = parse_result["success"]
                        expr_sympy = sp.sympify(parsed_expr)
                        
                        # FUNGSI LAMBDA
                        f = sp.lambdify(x, expr_sympy, modules=['numpy'])
                        
                        Y = []
                        valid_points = 0
                        
                        for x_point in X:
                            try:
                                y_val = f(x_point)
                                
                                # VALIDASI KETAT hasil
                                if (y_val is not None and 
                                    np.isfinite(y_val) and 
                                    not isinstance(y_val, (complex, sp.Basic)) and
                                    abs(y_val) < 1e10):
                                    Y.append(float(y_val))
                                    valid_points += 1
                                else:
                                    Y.append(np.nan)
                            except:
                                Y.append(np.nan)
                        
                        Y = np.array(Y)
                        
                        # PLOT JIKA ADA CUKUP TITIK
                        if valid_points > 30:
                            display_expr = ExpressionFormatter.format_display(expr_str)
                            
                            # Filter NaN values untuk plotting yang clean
                            x_clean = []
                            y_clean = []
                            
                            for x_val_pt, y_val_pt in zip(X, Y):
                                if not np.isnan(y_val_pt):
                                    x_clean.append(x_val_pt)
                                    y_clean.append(y_val_pt)
                            
                            if len(x_clean) > 10:
                                fig.add_trace(go.Scatter(
                                    x=x_clean, y=y_clean, mode='lines', 
                                    line=dict(color=colors[i], width=3),
                                    name=f'{label} = {display_expr}',
                                    connectgaps=False
                                ))
                                valid_plots += 1
                            
                    except Exception as e:
                        continue
            
            if valid_plots == 0:
                return PlottingEngine._no_plot_message("Tidak ada fungsi yang berhasil di-plot")
            
            # GARIS LIMIT & ANNOTATION
            fig.add_vline(x=x_val_float, line_dash="dash", line_color="red", 
                         opacity=0.7, annotation_text=f"x → {x_val}")
            
            # LAYOUT FINAL
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color=COLORS['text'], family=FONT, size=14),
                title=dict(
                    text="📈 Visualisasi 2D - Grafik Fungsi",
                    x=0.5,
                    font=dict(size=20, color=COLORS['text'])
                ),
                height=550,
                showlegend=True,
                xaxis_title="x",
                yaxis_title="f(x)",
                xaxis=dict(
                    range=[left_range, right_range],
                    gridcolor='lightgray',
                    zerolinecolor='lightgray'
                ),
                yaxis=dict(
                    gridcolor='lightgray', 
                    zerolinecolor='lightgray'
                ),
                legend=dict(
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='gray',
                    borderwidth=1
                )
            )
            
            return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            
        except Exception as e:
            return PlottingEngine._error_plot_message(f"Error membuat plot 2D: {str(e)}")

    @staticmethod
    def create_3d_plot(expr_str):
        try:
            if not expr_str or not expr_str.strip():
                return PlottingEngine._no_plot_message("Ekspresi kosong")
                
            x, y = sp.symbols('x y')
            
            parse_result = ExpressionParser.parse(expr_str)
            if "error" in parse_result:
                return PlottingEngine._no_plot_message(f"Ekspresi tidak valid: {parse_result['error']}")
            
            parsed_expr = parse_result["success"]
            expr = sp.sympify(parsed_expr)
            
            # CEK VARIABEL Y
            free_symbols = {str(s) for s in expr.free_symbols}
            if 'y' not in free_symbols:
                return PlottingEngine._no_plot_message("Fungsi tidak mengandung variabel y")
            
            # GRID 3D
            X_vals = np.linspace(-2.5, 2.5, 50)
            Y_vals = np.linspace(-2.5, 2.5, 50)
            X, Y = np.meshgrid(X_vals, Y_vals)
            
            # EVALUASI FUNGSI
            f = sp.lambdify((x, y), expr, modules=['numpy'])
            
            Z = np.zeros_like(X)
            valid_points = 0
            
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        z_val = f(X[i,j], Y[i,j])
                        if (z_val is not None and 
                            np.isfinite(z_val) and 
                            not isinstance(z_val, (complex, sp.Basic)) and
                            abs(z_val) < 100):
                            Z[i,j] = float(z_val)
                            valid_points += 1
                        else:
                            Z[i,j] = 0
                    except:
                        Z[i,j] = 0
            
            if valid_points < 500:
                return PlottingEngine._no_plot_message("Fungsi 3D tidak menghasilkan cukup titik valid")
            
            # BUAT PLOT 3D
            fig = go.Figure(data=[go.Surface(
                z=Z, x=X, y=Y, 
                colorscale='Viridis',
                opacity=0.9,
                showscale=True,
                surfacecolor=Z,
                contours=dict(
                    x=dict(show=True, color='gray', width=1),
                    y=dict(show=True, color='gray', width=1),
                    z=dict(show=True, color='gray', width=1)
                )
            )])
            
            fig.update_layout(
                title=dict(
                    text='🎨 Visualisasi 3D f(x,y)',
                    x=0.5,
                    font=dict(size=20, color=COLORS['text'])
                ),
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y', 
                    zaxis_title='f(x,y)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                    aspectmode='cube',
                    bgcolor='white'
                ),
                font=dict(family=FONT, color=COLORS['text'], size=12), 
                height=600,
                margin=dict(l=0, r=0, b=0, t=50),
                paper_bgcolor='white'
            )
            
            return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            
        except Exception as e:
            return PlottingEngine._error_plot_message(f"Error membuat plot 3D: {str(e)}")
    
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
            
            # Jika tidak terdefinisi, langsung return
            if not defined:
                analysis['steps'].append("❌ <b>f(a) tidak terdefinisi</b>")
                analysis.update({
                    'continuous': False, 
                    'discontinuity_type': "Diskontinu karena f(a) tidak terdefinisi"
                })
                return analysis
            
            limit_result_left = LimitCalculator.calculate(expr_str, x_val, 'left')
            limit_result_right = LimitCalculator.calculate(expr_str, x_val, 'right')
            limit_result_both = LimitCalculator.calculate(expr_str, x_val, 'both')
            
            analysis.update({
                'limit_left': limit_result_left.get('success') if 'success' in limit_result_left else limit_result_left.get('error', 'Error'),
                'limit_right': limit_result_right.get('success') if 'success' in limit_result_right else limit_result_right.get('error', 'Error'),
                'limit_both': limit_result_both.get('success') if 'success' in limit_result_both else limit_result_both.get('error', 'Error')
            })
            
            analysis['steps'].extend([
                f"📊 Limit kiri: {analysis['limit_left']}",
                f"📊 Limit kanan: {analysis['limit_right']}",
                f"📊 Limit kedua arah: {analysis['limit_both']}",
                f"📊 Nilai fungsi f({x_val}): {analysis['func_value']}"
            ])
            
            continuity_result = ContinuityAnalyzer._determine_continuity(analysis, x_val)
            analysis.update(continuity_result)
            
            # Tambahkan penjelasan detail
            explanation = ContinuityAnalyzer._get_detailed_explanation(analysis)
            analysis['steps'].append(f"💡 <b>Penjelasan:</b> {explanation}")
            
        except Exception as e:
            analysis['steps'].append(f"❌ Error dalam analisis: {str(e)}")
        
        return analysis
    
    @staticmethod
    def _check_definition(expr, x, x_val):
        if x_val in ['oo', '-oo']:
            return False, "Tidak terdefinisi"
        
        try:
            func_val = expr.subs(x, x_val)
            if func_val.is_infinite or func_val.is_complex or func_val == sp.nan or func_val == sp.zoo:
                return False, "Tidak terdefinisi"
            return True, LimitCalculator._format_result(func_val)
        except:
            return False, "Tidak terdefinisi"
    
    @staticmethod
    def _determine_continuity(analysis, x_val):
        # Cek apakah limit kiri dan kanan sama (dengan toleransi)
        limit_exists = ContinuityAnalyzer._limits_equal(analysis['limit_left'], analysis['limit_right'])
        
        # Cek apakah limit sama dengan nilai fungsi
        limit_equals_value = ContinuityAnalyzer._limits_equal(analysis['limit_both'], analysis['func_value'])
        
        if limit_exists and limit_equals_value:
            return {'continuous': True, 'discontinuity_type': "Kontinu"}
        else:
            if not limit_exists:
                return {'continuous': False, 'discontinuity_type': "Diskontinuitas lompatan"}
            else:
                return {'continuous': False, 'discontinuity_type': "Diskontinuitas yang dapat dihapus"}
    
    @staticmethod
    def _limits_equal(limit1, limit2):
        """Periksa apakah dua nilai limit sama (dengan toleransi untuk float)"""
        if limit1 == limit2:
            return True
        
        # Handle infinity cases
        if limit1 in ['∞', 'oo'] and limit2 in ['∞', 'oo']:
            return True
        if limit1 in ['-∞', '-oo'] and limit2 in ['-∞', '-oo']:
            return True
        
        # Coba konversi ke float untuk perbandingan numerik
        try:
            val1 = ContinuityAnalyzer._parse_limit_value(limit1)
            val2 = ContinuityAnalyzer._parse_limit_value(limit2)
            
            if val1 is not None and val2 is not None:
                # Pakai math.isclose() dengan toleransi
                return math.isclose(val1, val2, rel_tol=1e-9, abs_tol=1e-12)
        except:
            pass
        
        return False
    
    @staticmethod
    def _parse_limit_value(limit_str):
        """Parse string limit menjadi nilai float"""
        if limit_str in ['∞', 'oo']:
            return float('inf')
        elif limit_str in ['-∞', '-oo']:
            return float('-inf')
        elif limit_str in ['Tidak terdefinisi', 'Error menghitung limit:']:
            return None
        
        try:
            # Coba parse sebagai float langsung
            return float(limit_str)
        except:
            # Coba parse fraction
            if '/' in limit_str:
                try:
                    num, den = limit_str.split('/')
                    return float(num) / float(den)
                except:
                    pass
            return None
    
    @staticmethod
    def _get_detailed_explanation(analysis):
        """Berikan penjelasan detail tentang kontinuitas"""
        if analysis['continuous']:
            return "Fungsi kontinu karena limit sama dengan nilai fungsi di titik tersebut."
        
        if not analysis['defined']:
            return "Fungsi tidak terdefinisi di titik ini (misal: pembagian nol, akar negatif)."
        
        if analysis['discontinuity_type'] == "Diskontinuitas lompatan":
            return "Limit kiri dan kanan berbeda, menunjukkan lompatan pada grafik."
        elif analysis['discontinuity_type'] == "Diskontinuitas yang dapat dihapus":
            return "Limit ada tetapi tidak sama dengan nilai fungsi (biasanya hole pada grafik)."
        
        return "Fungsi memiliki diskontinuitas pada titik ini."

# ========== MODULE 6: UI COMPONENTS ==========
class ExpressionFormatter:
    @staticmethod
    def format_display(expr_str):
        if not expr_str:
            return ""
        
        expr_str = str(expr_str)
        
        # Step-by-step replacement yang lebih aman
        expr_str = re.sub(r'\*\*(\d+)', r'^\1', expr_str)
        
        replacements = [
            (r'sqrt\(', '√('),
            (r'pi', 'π'),
            (r'oo', '∞'),
            (r'exp\(', 'e^('),
            (r'asin', 'arcsin'),
            (r'acos', 'arccos'), 
            (r'atan', 'arctan'),
            (r'log', 'ln'),
            (r'\^2', '²'),
            (r'\^3', '³'),
        ]
        
        for old, new in replacements:
            expr_str = expr_str.replace(old, new)
        
        # Remove multiplication signs for cleaner display
        expr_str = re.sub(r'(\d)([a-zA-Z(])', r'\1\2', expr_str)
        expr_str = re.sub(r'([a-zA-Z)])(\d)', r'\1\2', expr_str)
        
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
        {"name": "∞ Limit Tak Hingga", "expr": "(2*x**2 + 3)/(x**2 - 1)", "xval": "infinity"},
        {"name": "√ Limit Akar", "expr": "sqrt(x-2)", "xval": "2"},
        {"name": "📈 Limit Eksponensial", "expr": "(1 + 1/x)**x", "xval": "infinity"},
        {"name": "🎭 Limit Piecewise", "expr": "(x**2 - 1)/(x - 1)", "xval": "1"},
        {"name": "🧊 Fungsi 3D", "expr": "x**2 + y**2", "xval": "0"}
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
    
    # Cek apakah fungsi 2 variabel
    x, y = sp.symbols('x y')
    parsed_expr = sp.sympify(parse_result["success"])
    free_symbols = {str(s) for s in parsed_expr.free_symbols}
    
    if 'y' in free_symbols:
        # Fungsi 2 variabel - skip continuity analysis
        continuity1 = {
            'steps': [
                "🔍 <b>Fungsi 2 Variabel Detected</b>",
                "📊 Analisis kontinuitas untuk fungsi multivariabel lebih kompleks",
                "🎨 Lihat visualisasi 3D untuk memahami perilaku fungsi"
            ],
            'continuous': None,
            'discontinuity_type': "Fungsi multivariabel - analisis lebih kompleks"
        }
    else:
        # Fungsi 1 variabel - lakukan analisis kontinuitas normal
        continuity1 = ContinuityAnalyzer.analyze(expr1, xval)
    
    results['function1'] = {
        'expression': expr1,
        'display_expression': ExpressionFormatter.format_display(expr1),
        'limit': limit_result1.get('success') if 'success' in limit_result1 else limit_result1.get('error'),
        'continuity': continuity1,
        'is_multivariable': 'y' in free_symbols
    }
    
    # Function 2 (if provided)
    if expr2 and expr2.strip():
        parse_result2 = ExpressionParser.parse(expr2)
        if "error" not in parse_result2:
            limit_result2 = LimitCalculator.calculate(expr2, xval, direction)
            parsed_expr2 = sp.sympify(parse_result2["success"])
            free_symbols2 = {str(s) for s in parsed_expr2.free_symbols}
            
            if 'y' in free_symbols2:
                continuity2 = {
                    'steps': ["🔍 Fungsi 2 Variabel - analisis kontinuitas skipped"],
                    'continuous': None,
                    'discontinuity_type': "Fungsi multivariabel"
                }
            else:
                continuity2 = ContinuityAnalyzer.analyze(expr2, xval)
            
            results['function2'] = {
                'expression': expr2,
                'display_expression': ExpressionFormatter.format_display(expr2),
                'limit': limit_result2.get('success') if 'success' in limit_result2 else limit_result2.get('error'),
                'continuity': continuity2,
                'is_multivariable': 'y' in free_symbols2
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
