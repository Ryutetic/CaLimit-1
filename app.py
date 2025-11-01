from flask import Flask, render_template, request, jsonify
from sympy import *
import traceback
import re

app = Flask(__name__)

# Define semua simbol yang bakal dipake
x, y, z = symbols('x y z')
pi = symbols('pi')
e = symbols('e')
deg = symbols('deg')

def konversi_ke_superscript(teks):
    """Convert x2 jadi x², x3 jadi x³, etc"""
    superscript_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
    }
    
    def replace_power(match):
        base = match.group(1)
        power = match.group(2)
        superscript_power = ''.join(superscript_map.get(char, char) for char in power)
        return base + superscript_power
    
    teks = re.sub(r'([a-zA-Z0-9\)])\^?(\d+)', replace_power, teks)
    return teks

def parse_fungsi(func_str):
    try:
        # Step 1: Convert superscript ke format sympy
        cleaned = func_str.replace('²', '**2').replace('³', '**3').replace('⁴', '**4')
        cleaned = cleaned.replace('⁵', '**5').replace('⁶', '**6').replace('⁷', '**7')
        cleaned = cleaned.replace('⁸', '**8').replace('⁹', '**9').replace('⁰', '**0')
        
        # Step 2: Handle simbol khusus
        cleaned = cleaned.replace('π', 'pi').replace('π', 'pi')
        cleaned = cleaned.replace('°', '*pi/180')  # Convert degree to radian
        cleaned = cleaned.replace('∞', 'oo')
        cleaned = cleaned.replace('^', '**')
        
        # Step 3: Handle fungsi trigonometri
        cleaned = re.sub(r'sin\(', 'sin(', cleaned)
        cleaned = re.sub(r'cos\(', 'cos(', cleaned)
        cleaned = re.sub(r'tan\(', 'tan(', cleaned)
        cleaned = re.sub(r'log\(', 'log(', cleaned)
        cleaned = re.sub(r'ln\(', 'ln(', cleaned)
        
        # Step 4: Handle perkalian implisit: 2x -> 2*x, x( -> x*(
        cleaned = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', cleaned)  # 2x -> 2*x
        cleaned = re.sub(r'([a-zA-Z0-9\)])(\()', r'\1*\2', cleaned)  # x( -> x*(
        
        print(f"Fungsi input: {func_str}")
        print(f"Fungsi cleaned: {cleaned}")
        
        # Parse dengan sympy
        expr = parse_expr(cleaned, transformations='all')
        print(f"Fungsi parsed: {expr}")
        
        return expr
        
    except Exception as e:
        print(f"Error parsing: {e}")
        raise ValueError(f"Fungsi tidak valid: {str(e)}")

def buat_penjelasan(operasi, fungsi, hasil, **kwargs):
    fungsi_tampil = konversi_ke_superscript(fungsi)
    hasil_tampil = konversi_ke_superscript(str(hasil))
    
    if operasi == 'limit':
        return f"""
MENGHITUNG LIMIT

Fungsi: {fungsi_tampil}
x mendekati: {kwargs['pendekatan']}

Langkah-langkah:
1. Substitusi x = {kwargs['pendekatan']}
2. Hitung nilai fungsi
3. Sederhanakan hasil

Hasil: lim({fungsi_tampil}) = {hasil_tampil}
Nilai numerik: {kwargs['nilai']}"""

    elif operasi == 'turunan':
        return f"""
MENCARI TURUNAN

Fungsi: {fungsi_tampil}

Langkah-langkah:
1. Turunkan pangkat ke koefisien
2. Kurangi pangkat dengan 1
3. Gabungkan semua suku

Turunan: {hasil_tampil}

Artinya: Setiap x berubah 1 satuan, fungsi berubah {hasil_tampil}"""

    elif operasi == 'integral_tentu':
        return f"""
MENGHITUNG INTEGRAL TENTU

Fungsi: {fungsi_tampil}
Batas: x = {kwargs['bawah']} sampai x = {kwargs['atas']}

Langkah-langkah:
1. Cari anti-turunan (integral tak tentu)
2. Hitung di batas atas (x={kwargs['atas']})
3. Hitung di batas bawah (x={kwargs['bawah']})
4. Kurangkan: F({kwargs['atas']}) - F({kwargs['bawah']})

Hasil: ∫({fungsi_tampil}) dx = {hasil_tampil}
Luas daerah: {kwargs['nilai']}"""

    elif operasi == 'integral_tak_tentu':
        return f"""
MENCARI INTEGRAL TAK TENTU

Fungsi: {fungsi_tampil}

Langkah-langkah:
1. Naikkan pangkat setiap suku
2. Bagi dengan pangkat baru
3. Tambahkan konstanta +C

Hasil: ∫({fungsi_tampil}) dx = {hasil_tampil} + C

Keterangan: +C menyatakan keluarga fungsi yang mungkin"""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/hitung', methods=['POST'])
def hitung():
    try:
        data = request.get_json()
        op = data['operasi']
        fungsi_str = data['fungsi']
        
        print(f"Memproses: {op} untuk fungsi: {fungsi_str}")
        
        fungsi = parse_fungsi(fungsi_str)

        if op == 'limit':
            pendekatan = data.get('pendekatan', '0')
            print(f"Menghitung limit mendekati: {pendekatan}")
            
            if pendekatan in ['∞', 'oo', 'inf']:
                hasil = limit(fungsi, x, oo)
            else:
                # Coba convert ke float, kalo gagal pake string
                try:
                    pendekatan_val = float(pendekatan)
                except:
                    pendekatan_val = pendekatan
                hasil = limit(fungsi, x, pendekatan_val)
            
            hasil_str = str(hasil).replace('**', '^').replace('*', '')
            try:
                nilai = float(hasil.evalf())
            except:
                nilai = str(hasil)
            
            print(f"Hasil limit: {hasil}")
            
            return jsonify({
                'sukses': True,
                'tipe': 'limit',
                'hasil': hasil_str,
                'nilai': nilai,
                'fungsi': fungsi_str,
                'pendekatan': pendekatan,
                'penjelasan': buat_penjelasan('limit', fungsi_str, hasil_str, 
                                            pendekatan=pendekatan, nilai=nilai)
            })

        elif op == 'turunan':
            print("Menghitung turunan...")
            hasil = diff(fungsi, x)
            hasil_str = str(hasil).replace('**', '^').replace('*', '')
            
            print(f"Hasil turunan: {hasil}")
            
            return jsonify({
                'sukses': True,
                'tipe': 'turunan', 
                'hasil': hasil_str,
                'fungsi': fungsi_str,
                'penjelasan': buat_penjelasan('turunan', fungsi_str, hasil_str)
            })

        elif op == 'integral':
            if 'bawah' in data and 'atas' in data:
                print(f"Menghitung integral tentu dari {data['bawah']} sampai {data['atas']}")
                # Integral tentu
                hasil_integral = integrate(fungsi, x)
                hasil_numerik = integrate(fungsi, (x, float(data['bawah']), float(data['atas'])))
                nilai = float(hasil_numerik.evalf())
                hasil_str = str(hasil_integral).replace('**', '^').replace('*', '')
                
                print(f"Hasil integral tentu: {hasil_integral}")
                
                return jsonify({
                    'sukses': True,
                    'tipe': 'integral_tentu',
                    'hasil': hasil_str,
                    'nilai': nilai,
                    'fungsi': fungsi_str,
                    'bawah': data['bawah'],
                    'atas': data['atas'],
                    'penjelasan': buat_penjelasan('integral_tentu', fungsi_str, hasil_str,
                                                bawah=data['bawah'], atas=data['atas'], nilai=nilai)
                })
            else:
                print("Menghitung integral tak tentu...")
                # Integral tak tentu
                hasil = integrate(fungsi, x)
                hasil_str = str(hasil).replace('**', '^').replace('*', '')
                
                print(f"Hasil integral tak tentu: {hasil}")
                
                return jsonify({
                    'sukses': True,
                    'tipe': 'integral_tak_tentu',
                    'hasil': hasil_str,
                    'fungsi': fungsi_str,
                    'penjelasan': buat_penjelasan('integral_tak_tentu', fungsi_str, hasil_str)
                })

    except Exception as e:
        print(f"Error: {traceback.format_exc()}")
        return jsonify({'sukses': False, 'error': str(e)})

if __name__ == '__main__':
    print("🐼 Kalkulator Limitra  jalan di http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
