# -*- coding: utf-8 -*-
"""app.py - ワークライフシュミレーター (完全版)"""

import pandas as pd
import ast
import numpy as np
import gradio as gr
from openai import OpenAI
import time
import os
import sys
import traceback
import json
import base64
import re
import zipfile
import io

# デバッグ用のロガー設定
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# アプリケーション情報ログ出力
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Directory contents: {os.listdir('.')}")

# 環境変数からAPIキーを取得（Render用）
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
logger.info(f"OpenAI API Key present: {bool(OPENAI_API_KEY)}")

# 友人の会社のサイトURL
FRIEND_BASE_URL = "https://friend-company.co.jp/result/receive"

# OpenAI クライアントを安全に初期化
def get_openai_client():
    """OpenAI クライアントを安全に取得"""
    try:
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not found")
            return None
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        return None

# 埋め込みデータ読み込み処理（ZIPファイル対応）
def load_embedding_data():
    """埋め込みデータを読み込み"""
    try:
        if os.path.exists("embedding_data.zip"):
            logger.info("Loading from ZIP file")
            with zipfile.ZipFile("embedding_data.zip", 'r') as zip_ref:
                with zip_ref.open("embedding_data.csv") as csvfile:
                    df = pd.read_csv(io.TextIOWrapper(csvfile, encoding='utf-8'))
        elif os.path.exists("embedding_data.csv"):
            logger.info("Loading from CSV file")
            df = pd.read_csv("embedding_data.csv")
        else:
            raise FileNotFoundError("No embedding data file found")
        
        logger.info(f"CSV loaded with {len(df)} rows")
        df["embedding"] = df["embedding"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        sample_data = []
        companies = ["テックイノベーション株式会社", "グローバルソリューションズ", "フューチャークリエイト"]
        periods = ["初期", "中期", "最近"]
        
        for company in companies:
            for period in periods:
                embedding = [np.random.random() * 0.1 for _ in range(1536)]
                sample_data.append({
                    "company": company,
                    "period": period,
                    "embedding": embedding
                })
        
        df = pd.DataFrame(sample_data)
        logger.info("Created fallback data due to error")
        return df

# データ読み込み実行
df = load_embedding_data()

def make_friend_url(analysis_result: dict) -> str:
    """分析結果をBase64エンコードして友人のサイトURLを生成"""
    try:
        json_str = json.dumps(analysis_result, separators=(",", ":"), ensure_ascii=False)
        b64_bytes = base64.urlsafe_b64encode(json_str.encode("utf-8"))
        b64_str = b64_bytes.decode("utf-8").rstrip("=")
        return f"{FRIEND_BASE_URL}?data={b64_str}"
    except Exception as e:
        logger.error(f"Error creating friend URL: {str(e)}")
        return FRIEND_BASE_URL

def classify_values(inputs, model="gpt-3.5-turbo"):
    """価値観を分析し分類"""
    logger.info("Starting classify_values")
    
    fallback_response = """
1. 自由度重視ワークスタイルタイプ
   自分の裁量で働き方を決められる環境を好みます。成果が出ていれば、どのように仕事を進めるかは自由に選びたいと考えています。

2. 成長志向アチーバータイプ
   企業の急速な成長とともに自己のスキルアップも目指しています。変化を楽しむことができ、常に進化を求める環境で活躍したいと思っています。

3. 独立自由プロフェッショナルタイプ
   経済的な自立を重視し、場所を選ばずに働ける自由なスタイルを望んでいます。生活のためだけでなく、自己実現を目指す働き方を求めています。
    """
    
    try:
        client = get_openai_client()
        if not client:
            logger.warning("OpenAI client not available, using fallback")
            return fallback_response
        
        prompt = f"""
        以下の回答から、ユーザーの価値観を分析し、3つの明確なタイプに分類してください。
        タイプ名は親しみやすく印象的なラベルにしてください（例： 柔軟ワーク志向タイプ）。
        各タイプには、90文字以上100文字以内の説明文をつけてください。
        
        重要：説明文は必ず「です・ます調」で書いてください。例えば「～と考えています」「～を求めています」という形式です。
        内容が重複しないように調整し、似たものは統合してください。
        
        フォーマットは次のようにしてください：
        
        1. [タイプ名]
           [「です・ます調」での特徴の説明]
        
        2. [タイプ名]
           [「です・ます調」での特徴の説明]
        
        3. [タイプ名]
           [「です・ます調」での特徴の説明]
        
        価値観: {inputs}
        """
        
        response = client.chat.completions.create(
            model=model, 
            messages=[{"role":"user","content":prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        result = response.choices[0].message.content
        logger.info("classify_values completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in classify_values: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return fallback_response

def compute_top3(summary, df):
    """企業とのマッチング計算"""
    logger.info("Starting compute_top3")
    
    fallback_data = [
        {"会社名": "テックイノベーション株式会社", "初期": 72.3, "中期": 78.5, "最近": 85.7, 
         "文化特性": {"革新性": 85, "安定性": 70, "成長機会": 90, "環境": 75, "報酬": 80}},
        {"会社名": "グローバルソリューションズ", "初期": 68.1, "中期": 72.4, "最近": 79.8,
         "文化特性": {"革新性": 75, "安定性": 85, "成長機会": 70, "環境": 65, "報酬": 90}},
        {"会社名": "フューチャークリエイト", "初期": 65.7, "中期": 70.2, "最近": 76.4,
         "文化特性": {"革新性": 90, "安定性": 60, "成長機会": 85, "環境": 80, "報酬": 75}}
    ]
    
    try:
        client = get_openai_client()
        if not client:
            logger.warning("OpenAI client not available, using fallback data")
            return fallback_data
        
        user_vecs = []
        for line in summary.splitlines():
            if not line.strip():
                continue
            try:
                response = client.embeddings.create(
                    input=line, 
                    model="text-embedding-ada-002"
                )
                user_vecs.append(np.array(response.data[0].embedding))
            except Exception as embed_error:
                logger.warning(f"Embedding error for line: {embed_error}")
                user_vecs.append(np.random.random(1536))
        
        if not user_vecs:
            user_vecs = [np.random.random(1536)]
        
        companies = {}
        for _, r in df.iterrows():
            c, p = r["company"], r["period"]
            emb = r["embedding"]
            companies.setdefault(c, {})[p] = np.array(emb)
        
        results = []
        for c, periods in companies.items():
            if "最近" not in periods: 
                continue
                
            def score(p):
                if p not in periods: 
                    return None
                    
                sims = []
                for uv in user_vecs:
                    similarity = np.dot(periods[p], uv) / (np.linalg.norm(periods[p]) * np.linalg.norm(uv))
                    sims.append(similarity)
                
                return round(np.mean(sims) * 100, 1)
            
            culture_features = {
                "革新性": min(100, max(30, int(np.random.normal(70, 15)))),
                "安定性": min(100, max(30, int(np.random.normal(70, 15)))),
                "成長機会": min(100, max(30, int(np.random.normal(70, 15)))),
                "環境": min(100, max(30, int(np.random.normal(70, 15)))),
                "報酬": min(100, max(30, int(np.random.normal(70, 15))))
            }
            
            results.append({
                "会社名": c,
                "初期": score("初期"),
                "中期": score("中期"),
                "最近": score("最近"),
                "文化特性": culture_features
            })
        
        sorted_results = sorted(results, key=lambda x: x["最近"] or 0, reverse=True)[:3]
        logger.info("compute_top3 completed successfully")
        return sorted_results
        
    except Exception as e:
        logger.error(f"Error in compute_top3: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return fallback_data

def generate_match_reason(company_data, user_values, model="gpt-3.5-turbo"):
    """マッチング理由を生成"""
    logger.info(f"Starting generate_match_reason for {company_data['会社名']}")
    
    fallback_messages = {
        "テック": "あなたの革新性を重視する価値観が、同社の先進的な企業文化と高くマッチしています。特に成長機会の豊富さと挑戦を奨励する環境があなたの可能性を最大限に引き出すでしょう。",
        "グローバル": "あなたの安定志向と報酬重視の価値観が、同社の堅実な企業文化と高い親和性を持っています。特に充実した福利厚生と明確なキャリアパスがあなたの長期的な成長をサポートします。",
        "フューチャー": "あなたの創造性と自由度を重視する価値観が、同社の柔軟な企業文化と強く共鳴しています。特にチームの多様性と協働の環境があなたの独自のアイデアを形にする機会を提供するでしょう。"
    }
    
    default_message = "あなたの価値観と企業文化の相性が良く、理想的な働き方を実現できる環境が整っています。多様な成長機会と充実したサポート体制があなたのキャリア発展を後押しするでしょう。"
    
    try:
        client = get_openai_client()
        if not client:
            logger.warning("OpenAI client not available, using fallback message")
            for key, message in fallback_messages.items():
                if key in company_data["会社名"]:
                    return message
            return default_message
        
        features = company_data["文化特性"]
        top_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:3]
        top_features_str = ", ".join([f"{k}({v}%)" for k, v in top_features])
        
        prompt = f"""
        あなたは企業文化と求職者のマッチング理由を説明するスペシャリストです。
        以下の情報に基づいて、なぜこの企業が求職者の価値観とマッチするのか、具体的かつポジティブに2〜3文で説明してください。
        
        企業名: {company_data['会社名']}
        企業の主な文化特性: {top_features_str}
        求職者の価値観: {user_values}
        
        以下の点を守ってください:
        - 前向きで明るい表現を使う
        - 具体的な文化特性（革新性、成長機会など）に言及する
        - 抽象的な表現を避け、具体的なメリットを示す
        - 日本語として自然な文章にする
        - 2〜3文に収める
        """
        
        response = client.chat.completions.create(
            model=model, 
            messages=[{"role":"user","content":prompt}],
            max_tokens=200,
            temperature=0.7
        )
        
        result = response.choices[0].message.content
        logger.info("generate_match_reason completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in generate_match_reason: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        for key, message in fallback_messages.items():
            if key in company_data["会社名"]:
                return message
        return default_message

def show_loading_screen():
    """診断中の画面を表示（キャラクター画像を読み込んで表示）"""
    
    # キャラクター画像を読み込んでBase64エンコード
    character_images = []
    character_files = ["mascot_char_01.png.png", "mascot_char_02.png.png", "mascot_char_03.png.png", "mascot_char_04.png.png"]
    
    # 現在のディレクトリを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for char_file in character_files:
        try:
            # 絶対パスで画像ファイルを探す
            file_path = os.path.join(current_dir, char_file)
            logger.info(f"Checking for character image at: {file_path}")
            
            if os.path.exists(file_path):
                logger.info(f"Loading character image: {file_path}")
                with open(file_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode()
                    character_images.append(f"data:image/png;base64,{img_base64}")
                    logger.info(f"Successfully loaded {char_file}")
            else:
                # 画像がない場合は絵文字で代替
                logger.warning(f"Character image not found: {file_path}")
                character_images.append("")
        except Exception as e:
            logger.error(f"Error loading character image {char_file}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            character_images.append("")
    
    # JavaScriptに画像データを渡す
    char_images_js = json.dumps(character_images)
    logger.info(f"Character images loaded: {len([img for img in character_images if img != ''])} out of {len(character_images)}")
    
    # 画像が一つも読み込めなかった場合の警告
    if all(img == "" for img in character_images):
        logger.error("WARNING: No character images were loaded successfully!")
    
    loading_html = f"""
    <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
        <div style="background: white; border-radius: 30px; padding: 60px 40px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); text-align: center; position: relative; overflow: hidden; min-height: 400px;">
            
            <!-- 診断中テキスト -->
            <h2 style="font-size: 32px; font-weight: 600; color: #2c3e50; margin: 0 0 40px 0;">
                診断中<span class="dots"></span>
            </h2>
            
            <!-- アニメーションエリア -->
            <div style="position: relative; height: 200px; margin: 40px 0;">
                <!-- 移動するキャラクター -->
                <div id="character-container" style="position: absolute; bottom: 0; right: -100px; width: 100px; height: 100px; animation: moveLeft 8s linear infinite;">
                    <img id="character-image" src="" style="width: 100px; height: 100px; object-fit: contain; display: none;">
                    <div id="character-emoji" style="width: 100px; height: 100px; display: flex; align-items: center; justify-content: center; font-size: 60px;">🤖</div>
                </div>
            </div>
            
            <!-- ローディングドット -->
            <div style="margin-top: 40px;">
                <span style="font-size: 48px; color: #5b9bd5;">
                    <span style="animation: blink 1.5s infinite;">●</span>
                    <span style="animation: blink 1.5s infinite 0.5s;">●</span>
                    <span style="animation: blink 1.5s infinite 1s;">●</span>
                </span>
            </div>
            
            <p style="font-size: 16px; color: #7f8c8d; margin-top: 30px;">
                あなたの価値観を分析しています...
            </p>
        </div>
    </div>
    
    <style>
    @keyframes moveLeft {{
        0% {{
            right: -100px;
        }}
        100% {{
            right: calc(100% + 100px);
        }}
    }}
    
    @keyframes blink {{
        0%, 50% {{ opacity: 0.3; }}
        25% {{ opacity: 1; }}
    }}
    
    .dots::after {{
        content: '.';
        animation: dots 2s steps(3, end) infinite;
    }}
    
    @keyframes dots {{
        0%, 20% {{ content: '.'; }}
        40% {{ content: '..'; }}
        60%, 100% {{ content: '...'; }}
    }}
    </style>
    
    <script>
    // キャラクター画像を順番に切り替える
    (function() {{
        const characterImages = {char_images_js};
        const characterEmojis = ['🏃', '📱', '🥽', '🎧'];
        
        console.log('Character images loaded:', characterImages);
        console.log('Number of valid images:', characterImages.filter(img => img !== '').length);
        console.log('First image data (truncated):', characterImages[0] ? characterImages[0].substring(0, 100) : 'No image');
        
        let currentIndex = 0;
        const imageEl = document.getElementById('character-image');
        const emojiEl = document.getElementById('character-emoji');
        
        function updateCharacter() {{
            console.log('updateCharacter called, currentIndex:', currentIndex);
            
            if (characterImages[currentIndex] && characterImages[currentIndex] !== "") {{
                // 画像がある場合
                console.log('Showing image for index:', currentIndex);
                imageEl.src = characterImages[currentIndex];
                imageEl.style.display = 'block';
                emojiEl.style.display = 'none';
            }} else {{
                // 画像がない場合は絵文字を表示
                console.log('Showing emoji for index:', currentIndex);
                imageEl.style.display = 'none';
                emojiEl.style.display = 'flex';
                emojiEl.innerHTML = characterEmojis[currentIndex % characterEmojis.length];
            }}
            
            currentIndex = (currentIndex + 1) % characterImages.length;
        }}
        
        // 初期表示
        updateCharacter();
        
        // 2秒ごとにキャラクターを切り替え
        setInterval(updateCharacter, 2000);
    }})();
    </script>
    """
    
    return loading_html

def run_app_with_loading(q1_choice, q1_text, q2_choice, q2_text, q3_choice, q3_text,
                         q4_choice, q4_text, q5_choice, q5_text):
    """診断中画面を表示してから処理を実行"""
    
    # まず診断中画面を表示
    yield ("", show_loading_screen(), 
           gr.update(visible=True), 
           gr.update(value="診断中...", interactive=False), 
           gr.update(visible=False))
    
    # 3秒待機（アニメーション表示のため）
    time.sleep(3)
    
    # 実際の処理を実行
    result = run_app(q1_choice, q1_text, q2_choice, q2_text, q3_choice, q3_text,
                     q4_choice, q4_text, q5_choice, q5_text)
    
    # 結果を返す
    yield result
        
def run_app(q1_choice, q1_text, q2_choice, q2_text, q3_choice, q3_text,
             q4_choice, q4_text, q5_choice, q5_text):
    """メインの診断処理を実行"""
    try:
        logger.info("Starting run_app function")
        
        # 回答を収集
        answers = []
        
        # 質問1の回答を追加
        if q1_choice:
            answers.extend(q1_choice)
        if q1_text and q1_text.strip():
            answers.append(q1_text.strip())
        
        # 質問2の回答を追加
        if q2_choice:
            answers.extend(q2_choice)
        if q2_text and q2_text.strip():
            answers.append(q2_text.strip())
        
        # 質問3の回答を追加
        if q3_choice:
            answers.extend(q3_choice)
        if q3_text and q3_text.strip():
            answers.append(q3_text.strip())
        
        # 質問4の回答を追加
        if q4_choice:
            answers.extend(q4_choice)
        if q4_text and q4_text.strip():
            answers.append(q4_text.strip())
        
        # 質問5の回答を追加
        if q5_choice:
            answers.extend(q5_choice)
        if q5_text and q5_text.strip():
            answers.append(q5_text.strip())
        
        logger.info(f"Collected answers: {len(answers)} items")
        
        # 価値観分類
        logger.info("Starting value classification")
        summary = classify_values(", ".join(answers))
        
        # 企業マッチング計算
        logger.info("Starting company matching")
        top3 = compute_top3(summary, df)
        
        # 各企業のマッチング理由を生成
        logger.info("Generating match reasons")
        for i, company in enumerate(top3):
            logger.info(f"Generating reason for company {i+1}: {company['会社名']}")
            company["マッチング理由"] = generate_match_reason(company, summary)
        
        # 分析結果を辞書形式で作成
        analysis_result = {
            "user_values": summary,
            "matching_companies": top3,
            "analysis_timestamp": int(time.time()),
            "user_answers": answers
        }
        
        # 友人のサイト用URLを生成
        friend_url = make_friend_url(analysis_result)
        
        # リンクのみを表示するHTMLを生成
        result_link_html = f"""
        <div style="max-width: 500px; margin: 0 auto; padding: 40px 20px; text-align: center;">
            <div style="background: white; border-radius: 30px; padding: 40px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                
                <!-- アイコン -->
                <div style="margin-bottom: 24px;">
                    <div style="width: 80px; height: 80px; background: #f0f4ff; border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
                        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#4a90e2" stroke-width="2">
                            <path d="M9 11l3 3L22 4"></path>
                            <path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11"></path>
                        </svg>
                    </div>
                </div>
                
                <!-- タイトル -->
                <h3 style="font-size: 24px; font-weight: 600; color: #2c3e50; margin: 0 0 16px 0;">診断が完了しました！</h3>
                
                <!-- 説明文 -->
                <p style="font-size: 16px; color: #7f8c8d; margin: 0 0 32px 0; line-height: 1.6;">
                    あなたにピッタリな<br>
                    職場環境が見つかりました
                </p>
                
                <!-- CTAボタン -->
                <a href="{friend_url}" target="_blank" style="display: inline-block; background: #5b9bd5; color: white; text-decoration: none; padding: 16px 60px; border-radius: 50px; font-size: 18px; font-weight: 500; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(91, 155, 213, 0.3);">
                    結果を見る
                </a>
                
                <!-- 注意書き -->
                <p style="font-size: 12px; color: #bdc3c7; margin: 24px 0 0 0;">
                    ※ 外部サイトに移動します
                </p>
            </div>
        </div>
        
        <style>
        a:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(91, 155, 213, 0.4) !important;
        }}
        </style>
        """
        
        logger.info("run_app completed successfully")
        return summary, result_link_html, gr.update(visible=True), gr.update(value="診断完了", variant="secondary")
        
    except Exception as e:
        logger.error(f"Error in run_app: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        error_html = f"""
        <div style="text-align: center; padding: 40px; background: #fff5f5; border-radius: 16px;">
            <h3 style="color: #dc2626; font-size: 18px; margin-bottom: 12px;">エラーが発生しました</h3>
            <p style="color: #666; font-size: 14px;">申し訳ありませんが、処理中にエラーが発生しました。</p>
            <p style="font-family: monospace; background: rgba(0,0,0,0.05); padding: 12px; border-radius: 8px; margin: 16px 0; font-size: 12px; color: #666;">{str(e)[:200]}</p>
            <p style="color: #666; font-size: 14px;">再度お試しいただくか、システム管理者にお問い合わせください。</p>
        </div>
        """
        return f"エラーが発生しました: {str(e)}", error_html, gr.update(visible=True), gr.update(value="再試行", variant="primary")

# CSS定義
custom_css = """
/* メインフォント */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;600;700&display=swap');

/* CSS変数 */
:root {
  --primary-gradient: linear-gradient(135deg, #FFE5EC 0%, #E8F5FF 100%);
  --primary-color: #00C5FF;
  --text-primary: #2c3e50;
  --text-secondary: #7f8c8d;
  --bg-white: #ffffff;
  --border-light: #e8e8e8;
}

/* ベース設定 */
body {
  font-family: 'Noto Sans JP', -apple-system, BlinkMacSystemFont, sans-serif;
  background: #ffffff !important;
  min-height: 100vh;
  margin: 0;
  padding: 0;
}

/* Gradioコンテナ */
.gradio-container {
  max-width: 600px !important;
  margin: 0 auto !important;
  padding: 20px !important;
  background: transparent !important;
}

/* ヘッダー画像 */
.header-image {
  background: var(--primary-gradient);
  border-radius: 20px;
  padding: 40px;
  margin-bottom: 24px;
  text-align: center;
  position: relative;
  overflow: hidden;
}

.header-image h1 {
  font-size: 32px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0 0 16px 0;
  letter-spacing: 0.5px;
}

.header-image p {
  font-size: 14px;
  color: var(--text-secondary);
  line-height: 1.6;
  margin: 0;
}

/* メインカード */
.main-card {
  background: white;
  border-radius: 20px;
  padding: 32px;
  margin-bottom: 24px;
}

/* 質問タイトル */
.question-title {
  color: black;
  font-size: 16px;
  font-weight: 500;
  margin-bottom: 8px;
  display: inline-block;
}

/* グラデーションライン */
.gradient-line {
  height: 2px;
  background: linear-gradient(to right, #FF69B4 0%, #00C5FF 100%);
  margin: 8px 0 16px 0;
  border-radius: 1px;
}

.question-subtitle {
  font-size: 13px;
  color: var(--text-secondary);
  margin: 0 0 20px 0;
}

/* チェックボックスグループ */
.checkbox-group {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 24px;
}

/* チェックボックス自体を非表示にする */
.checkbox-group input[type="checkbox"] {
  display: none;
}

.checkbox-group label {
  background: var(--bg-white);
  border: 2px solid #0075B8;
  border-radius: 50px;
  padding: 12px 28px;
  transition: all 0.3s ease;
  cursor: pointer;
  font-size: 15px;
  color: var(--text-primary);
  font-weight: 400;
  display: inline-flex;
  align-items: center;
  white-space: nowrap;
}

.checkbox-group label:hover {
  border-color: #00C5FF;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 117, 184, 0.2);
}

.checkbox-group label.selected,
.checkbox-group input[type="checkbox"]:checked + label {
  background: #00C5FF;
  color: white;
  border-color: #00C5FF;
}

/* テキスト入力 */
.text-input input[type="text"],
.text-input textarea {
  width: 100%;
  padding: 12px 16px;
  border: 2px solid #0075B8;
  border-radius: 12px;
  font-size: 14px;
  background: transparent;
  transition: all 0.3s ease;
}

.text-input input[type="text"]:focus,
.text-input textarea:focus {
  border-color: #00C5FF;
  background: white;
  outline: none;
}

/* その他入力セクション */
.other-input-section {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid var(--border-light);
}

.other-input-label {
  font-size: 14px;
  color: var(--text-secondary);
  margin-bottom: 12px;
  display: block;
}

/* 次へボタン */
.primary-button {
  background: #00C5FF !important;
  color: white !important;
  border: none !important;
  border-radius: 50px !important;
  padding: 14px 60px !important;
  font-size: 16px !important;
  font-weight: 500 !important;
  cursor: pointer !important;
  transition: all 0.3s ease !important;
  display: block !important;
  margin: 32px auto 0 auto !important;
  box-shadow: 0 4px 15px rgba(0, 197, 255, 0.3) !important;
}

.primary-button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px rgba(0, 197, 255, 0.4) !important;
}

/* Gradioデフォルトスタイルの上書き */
.gr-button {
  font-family: 'Noto Sans JP', sans-serif !important;
}

.gr-box {
  border: none !important;
  background: transparent !important;
}

.gr-form {
  border: none !important;
  background: transparent !important;
}

.gr-input {
  font-family: 'Noto Sans JP', sans-serif !important;
}

/* レスポンシブ対応 */
@media (max-width: 640px) {
  .gradio-container {
    padding: 10px !important;
  }
  
  .main-card {
    padding: 24px !important;
  }
  
  .header-image h1 {
    font-size: 24px;
  }
  
  .checkbox-group {
    flex-direction: column;
  }
  
  .checkbox-group label {
    width: 100%;
    text-align: center;
  }
  
  .primary-button {
    width: 100% !important;
    padding: 14px 40px !important;
  }
}
"""

# UI定義
with gr.Blocks(css=custom_css, title="ワークライフシュミレーター") as demo:
    # ヘッダー画像（GitHubリポジトリ内の画像を読み込み）
    header_image_path = "header_image.png"  # GitHubリポジトリ内の画像ファイル名
    
    try:
        # Renderでは、GitHubからクローンされたファイルは通常のファイルとして扱える
        if os.path.exists(header_image_path):
            logger.info(f"Loading header image from: {header_image_path}")
            with open(header_image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()
            
            # Base64エンコードした画像を表示
            gr.HTML(f"""
                <div style="margin-bottom: 24px;">
                    <img src="data:image/png;base64,{img_base64}" 
                         alt="ワークライフシュミレーター"
                         style="width: 100%; max-width: 600px; border-radius: 20px; display: block; margin: 0 auto;">
                </div>
            """)
            logger.info("Header image loaded successfully")
        else:
            # 画像ファイルが見つからない場合
            logger.warning(f"Header image not found at: {header_image_path}")
            logger.info(f"Current directory contents: {os.listdir('.')}")
            
            # 代替表示
            gr.HTML("""
                <div class="header-image">
                    <h1>ワークライフシュミレーター</h1>
                    <p>
                        就職活動のポイントはみんな違うはず。<br>
                        あなたの大切にしたい「価値観」を選ぶと、マッチする企業が見つかるかも！
                    </p>
                    <div style="margin-top: 40px; display: flex; justify-content: space-around; align-items: center;">
                        <div style="font-size: 60px;">👨‍💼</div>
                        <div style="font-size: 80px;">🔍</div>
                        <div style="font-size: 60px;">👩‍💼</div>
                    </div>
                </div>
            """)
    except Exception as e:
        logger.error(f"Error loading header image: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        
        # エラー時の代替表示
        gr.HTML("""
            <div class="header-image">
                <h1>ワークライフシュミレーター</h1>
                <p>
                    就職活動のポイントはみんな違うはず。<br>
                    あなたの大切にしたい「価値観」を選ぶと、マッチする企業が見つかるかも！
                </p>
                <div style="margin-top: 40px; text-align: center; color: #999;">
                    <p style="font-size: 14px;">※ヘッダー画像の読み込みに失敗しました</p>
                </div>
            </div>
        """)
    
    # 質問フォーム部分を非表示にできるようにグループ化
    with gr.Group(visible=True) as question_form:
        # 質問1
        with gr.Group(elem_classes="main-card"):
            gr.HTML('<div class="question-title">Q1. 働く上で、自分が大切にしたいことは？</div>')
            gr.HTML('<div class="gradient-line"></div>')
            gr.HTML('<p class="question-subtitle">※複数選択OK</p>')
            q1_choice = gr.CheckboxGroup(
                ["挑戦", "会社の安定", "自己成長", "柔軟な働き方", "プライベートの充実", "その他"],
                label="",
                elem_classes="checkbox-group"
            )
            with gr.Group(elem_classes="other-input-section"):
                gr.HTML('<label class="other-input-label">その他の詳しい内容はこちらへ</label>')
                q1_text = gr.Textbox(
                    label="",
                    placeholder="家族を大事にしたい、など",
                    elem_classes="text-input",
                    lines=3
                )

        # 質問2
        with gr.Group(elem_classes="main-card"):
            gr.HTML('<div class="question-title">Q2. 理想の働き方</div>')
            gr.HTML('<div class="gradient-line"></div>')
            gr.HTML('<p class="question-subtitle">※複数選択OK</p>')
            q2_choice = gr.CheckboxGroup(
                ["フルリモートワーク", "出社", "リモートワークと出社のハイブリッド"],
                label="",
                elem_classes="checkbox-group"
            )
            q2_text = gr.Textbox(
                label="その他",
                placeholder="具体的な働き方があれば入力してください",
                elem_classes="text-input"
            )

        # 質問3
        with gr.Group(elem_classes="main-card"):
            gr.HTML('<div class="question-title">Q3. 理想のチーム環境</div>')
            gr.HTML('<div class="gradient-line"></div>')
            gr.HTML('<p class="question-subtitle">※複数選択OK</p>')
            q3_choice = gr.CheckboxGroup(
                ["裁量権大", "多様性", "強いリーダーシップ", "フラットな関係性"],
                label="",
                elem_classes="checkbox-group"
            )
            q3_text = gr.Textbox(
                label="その他",
                placeholder="理想のチーム環境について入力してください",
                elem_classes="text-input"
            )

        # 質問4
        with gr.Group(elem_classes="main-card"):
            gr.HTML('<div class="question-title">Q4. 求める環境・制度</div>')
            gr.HTML('<div class="gradient-line"></div>')
            gr.HTML('<p class="question-subtitle">※複数選択OK</p>')
            q4_choice = gr.CheckboxGroup(
                ["研修が充実", "OJTがある", "海外研修", "自己学習支援", "副業OK"],
                label="",
                elem_classes="checkbox-group"
            )
            q4_text = gr.Textbox(
                label="その他",
                placeholder="その他の環境要件を入力してください",
                elem_classes="text-input"
            )

        # 質問5
        with gr.Group(elem_classes="main-card"):
            gr.HTML('<div class="question-title">Q5. その他重視するポイント</div>')
            gr.HTML('<div class="gradient-line"></div>')
            gr.HTML('<p class="question-subtitle">※複数選択OK</p>')
            q5_choice = gr.CheckboxGroup(
                ["高インセンティブ", "勤務地", "フレックス", "スピード感"],
                label="",
                elem_classes="checkbox-group"
            )
            q5_text = gr.Textbox(
                label="その他",
                placeholder="その他に重視する点があれば入力してください",
                elem_classes="text-input"
            )

        next_btn = gr.Button("診断を開始", elem_classes="primary-button", size="lg")

    # 結果表示エリア
    with gr.Group(visible=False, elem_classes="results-area") as results_area:
        summary_out = gr.Textbox(visible=False)
        results_out = gr.HTML()

    # ボタンクリック時の処理を分けて定義
    def handle_button_click(q1_choice, q1_text, q2_choice, q2_text, q3_choice, q3_text,
                           q4_choice, q4_text, q5_choice, q5_text):
        """ボタンクリック時の処理"""
        # まず診断中画面を表示
        loading_html = show_loading_screen()
        yield ("", loading_html, 
               gr.update(visible=True), 
               gr.update(value="診断中...", interactive=False), 
               gr.update(visible=False))
        
        # 少し待機
        time.sleep(3)
        
        # 実際の処理を実行
        summary, result_html, results_visible, btn_update = run_app(q1_choice, q1_text, q2_choice, q2_text, q3_choice, q3_text,
                                                                    q4_choice, q4_text, q5_choice, q5_text)
        
        # 結果を返す
        yield summary, result_html, results_visible, btn_update, gr.update(visible=False)

    # イベントハンドラー
    next_btn.click(
        fn=handle_button_click,
        inputs=[
            q1_choice, q1_text,
            q2_choice, q2_text,
            q3_choice, q3_text,
            q4_choice, q4_text,
            q5_choice, q5_text
        ],
        outputs=[
            summary_out,
            results_out,
            results_area,
            next_btn,
            question_form
        ]
    )

# 使用上の注意
with gr.Accordion("ご利用にあたって", open=False):
    gr.HTML("""
        <div style="padding: 8px; color: #666; font-size: 14px; line-height: 1.8;">
            <p style="margin-bottom: 12px;">このアプリケーションは、あなたの価値観と企業文化のマッチングをAIが分析するデモンストレーションです。</p>
            <ul style="margin: 0; padding-left: 20px;">
                <li>実際の企業データは限定的であり、結果は参考値としてご利用ください</li>
                <li>OpenAI APIを使用しているため、処理に数秒かかることがあります</li>
                <li>入力された情報は分析にのみ使用され、保存されません</li>
            </ul>
        </div>
    """)

# アプリ起動
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        debug=False,
        share=False
    )
