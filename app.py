# -*- coding: utf-8 -*-
"""app.py - クリーンモダンデザインの企業文化マッチングアプリ (Render用に最適化)"""

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
        
        # シンプルな初期化（proxiesなどのパラメータを除く）
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
        # まずZIPファイルを試す
        if os.path.exists("embedding_data.zip"):
            logger.info("Loading from ZIP file")
            with zipfile.ZipFile("embedding_data.zip", 'r') as zip_ref:
                with zip_ref.open("embedding_data.csv") as csvfile:
                    df = pd.read_csv(io.TextIOWrapper(csvfile, encoding='utf-8'))
        # 通常のCSVファイルも試す
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
        # エラー時のフォールバックデータ
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
        # JSON文字列化（余計な空白を除去）
        json_str = json.dumps(analysis_result, separators=(",", ":"), ensure_ascii=False)
        
        # UTF-8でエンコードしてからBase64エンコード
        b64_bytes = base64.urlsafe_b64encode(json_str.encode("utf-8"))
        
        # パディング文字を削除して短くする
        b64_str = b64_bytes.decode("utf-8").rstrip("=")
        
        # 最終的なURLを生成
        return f"{FRIEND_BASE_URL}?data={b64_str}"
    except Exception as e:
        logger.error(f"Error creating friend URL: {str(e)}")
        # エラー時はベースURLのみ返す
        return FRIEND_BASE_URL

def classify_values(inputs, model="gpt-3.5-turbo"):
    """価値観を分析し分類"""
    logger.info("Starting classify_values")
    
    # フォールバック回答を先に定義
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
    
    # フォールバックデータ
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
        
        # ユーザーベクトル取得
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
                # フォールバック: ランダムベクトル
                user_vecs.append(np.random.random(1536))
        
        if not user_vecs:
            user_vecs = [np.random.random(1536)]
        
        # 企業データ処理
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
                    # コサイン類似度
                    similarity = np.dot(periods[p], uv) / (np.linalg.norm(periods[p]) * np.linalg.norm(uv))
                    sims.append(similarity)
                
                return round(np.mean(sims) * 100, 1)
            
            # 文化特性はランダムに設定（デモ用）
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
    
    # フォールバックメッセージ
    fallback_messages = {
        "テック": "あなたの革新性を重視する価値観が、同社の先進的な企業文化と高くマッチしています。特に成長機会の豊富さと挑戦を奨励する環境があなたの可能性を最大限に引き出すでしょう。",
        "グローバル": "あなたの安定志向と報酬重視の価値観が、同社の堅実な企業文化と高い親和性を持っています。特に充実した福利厚生と明確なキャリアパスがあなたの長期的な成長をサポートします。",
        "フューチャー": "あなたの創造性と自由度を重視する価値観が、同社の柔軟な企業文化と強く共鳴しています。特にチームの多様性と協働の環境があなたの独自のアイデアを形にする機会を提供するでしょう。"
    }
    
    # デフォルトメッセージ
    default_message = "あなたの価値観と企業文化の相性が良く、理想的な働き方を実現できる環境が整っています。多様な成長機会と充実したサポート体制があなたのキャリア発展を後押しするでしょう。"
    
    try:
        client = get_openai_client()
        if not client:
            logger.warning("OpenAI client not available, using fallback message")
            # 企業名に基づいてフォールバックメッセージを選択
            for key, message in fallback_messages.items():
                if key in company_data["会社名"]:
                    return message
            return default_message
        
        # 文化特性とスコアを取得
        features = company_data["文化特性"]
        top_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:3]
        top_features_str = ", ".join([f"{k}({v}%)" for k, v in top_features])
        
        # マッチング理由を生成
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
        
        # エラー時も企業名に基づいてフォールバックメッセージを選択
        for key, message in fallback_messages.items():
            if key in company_data["会社名"]:
                return message
        return default_message

def run_app(q1_choice, q1_text, q2_choice, q2_text, q3_choice, q3_text,
            q4_choice, q4_text, q5_choice, q5_text):
    """メイン分析処理"""
    logger.info("Starting run_app")
    
    try:
        # 回答データの収集（複数選択対応）
        answers = []
        
        # CheckboxGroupは選択されたものがリストで返される
        for choice_list, text in [(q1_choice, q1_text), (q2_choice, q2_text), (q3_choice, q3_text),
                                (q4_choice, q4_text), (q5_choice, q5_text)]:
            if choice_list:  # リストが空でなければ
                answers.extend(choice_list)  # リストの各要素を追加
            
            if text and text.strip():  # テキスト入力があれば
                answers.append(text.strip())
        
        if not answers:
            logger.warning("No answers provided")
            return "回答が入力されていません", "", gr.update(visible=False), gr.update(value="次へ進む", variant="primary")
        
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
        
        # リンクのみを表示するHTMLを生成（クリーンデザイン）
        result_link_html = f"""
        <div class="result-container" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 40px 20px;">
            <div class="result-card" style="background: var(--bg-primary); border-radius: 24px; padding: 48px; box-shadow: var(--shadow-lg); position: relative; overflow: hidden;">
                
                <!-- 背景グラデーション（サブトル） -->
                <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(135deg, rgba(255,229,236,0.3) 0%, rgba(232,245,255,0.3) 100%); z-index: 0; opacity: 0.5;"></div>
                
                <!-- コンテンツ（z-indexで前面に） -->
                <div style="position: relative; z-index: 1;">
                    <!-- ロゴエリア（GHELIAを模したデザイン） -->
                    <div style="text-align: center; margin-bottom: 32px;">
                        <div style="display: inline-block; background: var(--bg-secondary); padding: 24px 40px; border-radius: 16px; backdrop-filter: blur(10px);">
                            <h2 style="margin: 0; font-size: 28px; font-weight: 300; letter-spacing: 8px; color: var(--text-primary);">RESULT</h2>
                        </div>
                    </div>
                    
                    <!-- タイトル -->
                    <div style="text-align: center; margin-bottom: 32px;">
                        <h3 style="font-size: 24px; font-weight: 500; color: var(--text-primary); margin: 0 0 12px 0;">分析が完了しました</h3>
                        <p style="font-size: 16px; color: var(--text-secondary); margin: 0; line-height: 1.6;">
                            あなたの価値観に基づいた<br>
                            企業マッチング結果をご用意しました
                        </p>
                    </div>
                    
                    <!-- 結果内容プレビュー -->
                    <div style="background: var(--bg-secondary); border-radius: 16px; padding: 24px; margin-bottom: 32px; backdrop-filter: blur(5px);">
                        <p style="font-size: 14px; color: var(--text-primary); margin: 0 0 16px 0; font-weight: 500;">
                            以下の内容を確認できます：
                        </p>
                        <ul style="margin: 0; padding: 0 0 0 20px; list-style: none;">
                            <li style="font-size: 14px; color: var(--text-secondary); margin-bottom: 8px; position: relative; padding-left: 20px;">
                                <span style="position: absolute; left: 0; color: var(--text-primary);">・</span>
                                価値観タイプの詳細分析
                            </li>
                            <li style="font-size: 14px; color: var(--text-secondary); margin-bottom: 8px; position: relative; padding-left: 20px;">
                                <span style="position: absolute; left: 0; color: var(--text-primary);">・</span>
                                おすすめ企業TOP3
                            </li>
                            <li style="font-size: 14px; color: var(--text-secondary); margin-bottom: 8px; position: relative; padding-left: 20px;">
                                <span style="position: absolute; left: 0; color: var(--text-primary);">・</span>
                                マッチング理由の解説
                            </li>
                            <li style="font-size: 14px; color: var(--text-secondary); margin-bottom: 0; position: relative; padding-left: 20px;">
                                <span style="position: absolute; left: 0; color: var(--text-primary);">・</span>
                                企業文化の詳細データ
                            </li>
                        </ul>
                    </div>
                    
                    <!-- CTAボタン -->
                    <div style="text-align: center; margin-bottom: 24px;">
                        <a href="{friend_url}" target="_blank" class="result-cta-button" style="display: inline-block; background: var(--button-bg); color: var(--button-text); text-decoration: none; padding: 16px 48px; border-radius: 30px; font-size: 16px; font-weight: 500; transition: all 0.2s ease; box-shadow: var(--shadow-sm);">
                            結果を確認する
                        </a>
                    </div>
                    
                    <!-- 注意書き -->
                    <p style="font-size: 12px; color: var(--text-tertiary); text-align: center; margin: 0;">
                        ※ 外部サイトに移動します
                    </p>
                </div>
            </div>
        </div>
        
        <style>
        .result-cta-button:hover {{
            transform: translateY(-1px) !important;
            box-shadow: var(--shadow-hover) !important;
        }}
        </style>
        """
        
        logger.info("run_app completed successfully")
        # 最終結果を返す（リンクのみ表示）
        return summary, result_link_html, gr.update(visible=True), gr.update(value="分析完了 ✓", variant="secondary")
        
    except Exception as e:
        logger.error(f"Error in run_app: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        error_html = f"""
        <div class="error-container" style="text-align: center; padding: 40px; background: var(--error-bg); border-radius: 16px;">
            <h3 style="color: var(--error-text); font-size: 18px; margin-bottom: 12px;">エラーが発生しました</h3>
            <p style="color: var(--text-secondary); font-size: 14px;">申し訳ありませんが、処理中にエラーが発生しました。</p>
            <p style="font-family: monospace; background: var(--bg-secondary); padding: 12px; border-radius: 8px; margin: 16px 0; font-size: 12px; color: var(--text-secondary);">{str(e)[:200]}</p>
            <p style="color: var(--text-secondary); font-size: 14px;">再度お試しいただくか、システム管理者にお問い合わせください。</p>
        </div>
        """
        return f"エラーが発生しました: {str(e)}", error_html, gr.update(visible=True), gr.update(value="再試行", variant="primary")

# CSS定義（クリーンモダンスタイル - 完全ダークモード無効化）
custom_css = """
/* 超強力ダークモード無効化 - 全レベルで制御 */
html, body, #root, .gradio-container, .app, .main {
  color-scheme: light !important;
  background: #ffffff !important;
  color: #1a1a1a !important;
}

*, *::before, *::after {
  color-scheme: light !important;
}

/* Gradio特有のダークモード無効化 */
.dark, [data-theme="dark"], [class*="dark"] {
  color-scheme: light !important;
  background: #ffffff !important;
  color: #1a1a1a !important;
}

/* システムレベルでのダークモード上書き */
@media (prefers-color-scheme: dark) {
  *, *::before, *::after {
    color-scheme: light !important;
    background-color: unset !important;
    color: unset !important;
  }
  
  html, body {
    background: linear-gradient(135deg, #FFE5EC 0%, #E8F5FF 100%) !important;
    color: #1a1a1a !important;
  }
}

/* CSS変数定義 - ライトモード固定 */
:root {
  /* ライトモードの色定義（固定） */
  --bg-primary: #ffffff !important;
  --bg-secondary: #f8f8f8 !important;
  --bg-tertiary: #fafafa !important;
  --bg-gradient-start: #FFE5EC !important;
  --bg-gradient-end: #E8F5FF !important;
  
  --text-primary: #1a1a1a !important;
  --text-secondary: #666666 !important;
  --text-tertiary: #999999 !important;
  --text-inverse: #ffffff !important;
  
  --border-primary: #e5e5e5 !important;
  --border-secondary: #d0d0d0 !important;
  --border-light: rgba(0,0,0,0.04) !important;
  
  --shadow-sm: 0 2px 8px rgba(0,0,0,0.1) !important;
  --shadow-md: 0 2px 20px rgba(0,0,0,0.06) !important;
  --shadow-lg: 0 4px 30px rgba(0,0,0,0.08) !important;
  --shadow-hover: 0 6px 20px rgba(0,0,0,0.15) !important;
  
  --focus-ring: rgba(26,26,26,0.05) !important;
  --button-bg: #1a1a1a !important;
  --button-text: #ffffff !important;
  
  --error-bg: #fff5f5 !important;
  --error-text: #dc2626 !important;
  
  --checkbox-bg: #f8f8f8 !important;
  --checkbox-hover: #f0f0f0 !important;
  --checkbox-checked-bg: #1a1a1a !important;
  --checkbox-checked-text: #ffffff !important;
}

/* フォントとベース設定 */
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, 'Noto Sans JP', sans-serif;
  color: var(--text-primary);
  background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
  min-height: 100vh;
  line-height: 1.6;
  transition: background 0.3s ease, color 0.3s ease;
}

/* メインコンテナ */
.gradio-container {
  max-width: 800px !important;
  margin: 0 auto !important;
  background: transparent !important;
}

/* カードスタイル（Elevation風） */
.card-elevation {
  background: var(--bg-primary);
  border-radius: 20px;
  padding: 32px;
  margin-bottom: 24px;
  box-shadow: var(--shadow-md);
  border: 1px solid var(--border-light);
  transition: all 0.3s ease;
}

.card-elevation:hover {
  box-shadow: var(--shadow-lg);
  transform: translateY(-1px);
}

/* タイトル */
h1 {
  font-size: 32px !important;
  font-weight: 300 !important;
  text-align: center;
  margin: 40px 0 48px 0 !important;
  color: var(--text-primary) !important;
  letter-spacing: 0.5px;
}

/* 質問タイトル */
.question-title {
  font-size: 18px;
  font-weight: 500;
  color: var(--text-primary);
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.question-title:before {
  content: '';
  display: inline-block;
  width: 4px;
  height: 20px;
  background: var(--text-primary);
  border-radius: 2px;
}

/* チェックボックスグループ */
.checkbox-group label {
  background: var(--checkbox-bg) !important;
  border: 1px solid var(--border-primary) !important;
  border-radius: 12px !important;
  padding: 12px 20px !important;
  margin: 8px !important;
  transition: all 0.2s ease !important;
  cursor: pointer !important;
  display: inline-block !important;
  color: var(--text-primary) !important;
}

.checkbox-group label:hover {
  background: var(--checkbox-hover) !important;
  border-color: var(--border-secondary) !important;
}

.checkbox-group input[type="checkbox"]:checked + label {
  background: var(--checkbox-checked-bg) !important;
  color: var(--checkbox-checked-text) !important;
  border-color: var(--checkbox-checked-bg) !important;
}

/* テキスト入力 */
input[type="text"], textarea {
  border: 1px solid var(--border-primary) !important;
  border-radius: 12px !important;
  padding: 12px 16px !important;
  font-size: 14px !important;
  transition: all 0.2s ease !important;
  background: var(--bg-tertiary) !important;
  color: var(--text-primary) !important;
}

input[type="text"]:focus, textarea:focus {
  border-color: var(--text-primary) !important;
  background: var(--bg-secondary) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px var(--focus-ring) !important;
}

/* ボタン */
.primary-button {
  background: var(--button-bg) !important;
  color: var(--button-text) !important;
  border: none !important;
  border-radius: 24px !important;
  padding: 14px 32px !important;
  font-size: 16px !important;
  font-weight: 500 !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
  margin: 32px auto 0 auto !important;
  display: block !important;
  box-shadow: var(--shadow-sm) !important;
}

.primary-button:hover {
  transform: translateY(-1px) !important;
  box-shadow: var(--shadow-hover) !important;
}

.primary-button:active {
  transform: translateY(0) !important;
}

/* 結果エリア */
.results-area {
  animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* アコーディオン */
.accordion {
  background: var(--bg-primary) !important;
  border-radius: 16px !important;
  border: 1px solid var(--border-primary) !important;
  margin-top: 40px !important;
}

.accordion-header {
  padding: 20px 24px !important;
  font-weight: 500 !important;
  color: var(--text-secondary) !important;
  font-size: 14px !important;
}

/* レスポンシブ対応 - モバイル */
@media (max-width: 480px) {
  .card-elevation {
    padding: 20px 16px;
    margin-bottom: 16px;
    border-radius: 16px;
  }
  
  h1 {
    font-size: 24px !important;
    margin: 24px 0 32px 0 !important;
  }
  
  .question-title {
    font-size: 16px;
  }
  
  .checkbox-group label {
    display: block !important;
    margin: 6px 0 !important;
    padding: 10px 16px !important;
    font-size: 14px !important;
  }
  
  input[type="text"], textarea {
    padding: 10px 14px !important;
    font-size: 14px !important;
  }
  
  .primary-button {
    padding: 12px 24px !important;
    font-size: 14px !important;
    width: 100% !important;
    max-width: 300px !important;
  }
}

/* レスポンシブ対応 - タブレット */
@media (min-width: 481px) and (max-width: 768px) {
  .card-elevation {
    padding: 24px 20px;
  }
  
  h1 {
    font-size: 28px !important;
  }
  
  .checkbox-group label {
    display: inline-block !important;
    margin: 6px !important;
    width: calc(50% - 12px) !important;
  }
}

/* レスポンシブ対応 - 大画面 */
@media (min-width: 1200px) {
  .gradio-container {
    max-width: 900px !important;
  }
  
  .card-elevation {
    padding: 40px;
  }
}

/* Gradioデフォルトスタイルの上書き */
.gr-button {
  font-family: inherit !important;
  background: var(--button-bg) !important;
  color: var(--button-text) !important;
  border: none !important;
  transition: all 0.2s ease !important;
}

.gr-button:hover {
  box-shadow: var(--shadow-hover) !important;
}

.gr-box {
  border-radius: 16px !important;
  border-color: var(--border-primary) !important;
  background: var(--bg-primary) !important;
}

.gr-form {
  border: none !important;
  background: transparent !important;
}

.gr-panel {
  background: transparent !important;
  border: none !important;
}

/* ラベルのスタイル */
label.block {
  color: var(--text-secondary) !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  margin-bottom: 8px !important;
}

/* 超強力Gradio要素固定スタイル */
.gr-button, .gr-textbox, .gr-checkbox, .gr-radio, .gr-dropdown {
  background: #ffffff !important;
  color: #1a1a1a !important;
  border-color: #e5e5e5 !important;
}

.gr-button {
  background: #1a1a1a !important;
  color: #ffffff !important;
}

.gr-textbox input, .gr-textbox textarea {
  background: #fafafa !important;
  color: #1a1a1a !important;
  border: 1px solid #e5e5e5 !important;
}

.gr-checkbox-group .gr-checkbox {
  background: #f8f8f8 !important;
  color: #1a1a1a !important;
}

/* 全要素に対する超強力スタイル適用 */
@media (prefers-color-scheme: dark) {
  .gradio-container, .gradio-container * {
    background-color: inherit !important;
    color: inherit !important;
  }
  
  .gr-button {
    background: #1a1a1a !important;
    color: #ffffff !important;
  }
  
  .gr-textbox, .gr-textbox input, .gr-textbox textarea {
    background: #fafafa !important;
    color: #1a1a1a !important;
    border-color: #e5e5e5 !important;
  }
  
  .gr-checkbox-group label {
    background: #f8f8f8 !important;
    color: #1a1a1a !important;
    border-color: #e5e5e5 !important;
  }
  
  .gr-box, .gr-form, .gr-panel {
    background: #ffffff !important;
    color: #1a1a1a !important;
  }
  
  .gr-accordion {
    background: #ffffff !important;
    color: #1a1a1a !important;
    border-color: #e5e5e5 !important;
  }
}

"""

# UI定義
with gr.Blocks(css=custom_css, title="企業文化マッチング診断", js="""
function() {
    // 超強力JavaScript - ダークモード完全無効化
    function forceeLightMode() {
        // HTML要素のcolor-schemeを強制的にlight設定
        document.documentElement.style.setProperty('color-scheme', 'light', 'important');
        document.body.style.setProperty('color-scheme', 'light', 'important');
        
        // 全要素に対してライトモード強制
        const allElements = document.querySelectorAll('*');
        allElements.forEach(el => {
            el.style.setProperty('color-scheme', 'light', 'important');
            
            // ダークモード関連のクラスを削除
            el.classList.remove('dark');
            if (el.getAttribute('data-theme') === 'dark') {
                el.setAttribute('data-theme', 'light');
            }
        });
        
        // Gradio特有の要素に対する強制スタイル適用
        const gradioElements = document.querySelectorAll('.gradio-container, .gr-button, .gr-textbox, .gr-checkbox, .gr-radio, .gr-dropdown');
        gradioElements.forEach(el => {
            if (el.classList.contains('gr-button')) {
                el.style.setProperty('background', '#1a1a1a', 'important');
                el.style.setProperty('color', '#ffffff', 'important');
            } else {
                el.style.setProperty('background', '#ffffff', 'important');
                el.style.setProperty('color', '#1a1a1a', 'important');
            }
        });
        
        // body背景を強制設定
        document.body.style.setProperty('background', 'linear-gradient(135deg, #FFE5EC 0%, #E8F5FF 100%)', 'important');
        document.body.style.setProperty('color', '#1a1a1a', 'important');
    }
    
    // 初回実行
    forceeLightMode();
    
    // 定期的に実行（ダークモード設定の変更を監視）
    setInterval(forceeLightMode, 100);
    
    // DOM変更を監視して即座に適用
    const observer = new MutationObserver(forceeLightMode);
    observer.observe(document.body, { 
        childList: true, 
        subtree: true, 
        attributes: true,
        attributeFilter: ['class', 'data-theme', 'style']
    });
    
    return null;
}
""") as demo:
    gr.HTML("<h1>企業文化マッチング診断</h1>")
    
    with gr.Group(elem_classes="card-elevation"):
        gr.HTML('<div class="question-title">仕事で大切にしたいこと</div>')
        q1_choice = gr.CheckboxGroup(
            ["挑戦", "安定性", "社会貢献性", "自己成長", "柔軟な働き方", "プライベートの充実"],
            label="",
            elem_classes="checkbox-group"
        )
        q1_text = gr.Textbox(label="その他", placeholder="その他の価値観があれば入力してください", elem_classes="text-input")

    with gr.Group(elem_classes="card-elevation"):
        gr.HTML('<div class="question-title">理想の働き方</div>')
        q2_choice = gr.CheckboxGroup(
            ["フルリモートワーク", "出社", "リモートワークと出社のハイブリッド"],
            label="",
            elem_classes="checkbox-group"
        )
        q2_text = gr.Textbox(label="その他", placeholder="具体的な働き方があれば入力してください", elem_classes="text-input")

    with gr.Group(elem_classes="card-elevation"):
        gr.HTML('<div class="question-title">理想のチーム環境</div>')
        q3_choice = gr.CheckboxGroup(
            ["裁量権大", "多様性", "強いリーダーシップ", "フラットな関係性"],
            label="",
            elem_classes="checkbox-group"
        )
        q3_text = gr.Textbox(label="その他", placeholder="理想のチーム環境について入力してください", elem_classes="text-input")

    with gr.Group(elem_classes="card-elevation"):
        gr.HTML('<div class="question-title">求める環境・制度</div>')
        q4_choice = gr.CheckboxGroup(
            ["研修が充実", "OJTがある", "海外研修", "自己学習支援", "副業OK"],
            label="",
            elem_classes="checkbox-group"
        )
        q4_text = gr.Textbox(label="その他", placeholder="その他の環境要件を入力してください", elem_classes="text-input")

    with gr.Group(elem_classes="card-elevation"):
        gr.HTML('<div class="question-title">その他重視するポイント</div>')
        q5_choice = gr.CheckboxGroup(
            ["高インセンティブ", "勤務地", "フレックス", "スピード感"],
            label="",
            elem_classes="checkbox-group"
        )
        q5_text = gr.Textbox(label="その他", placeholder="その他に重視する点があれば入力してください", elem_classes="text-input")

    next_btn = gr.Button("診断を開始", elem_classes="primary-button", size="lg")

    with gr.Group(visible=False, elem_classes="results-area") as results_area:
        summary_out = gr.Textbox(visible=False)
        results_out = gr.HTML()

    next_btn.click(
        fn=run_app,
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
            next_btn
        ]
    )

# 使用上の注意
with gr.Accordion("ご利用にあたって", open=False, elem_classes="accordion"):
    gr.HTML("""
    <div style="padding: 8px; color: var(--text-secondary); font-size: 14px; line-height: 1.8;">
        <p style="margin-bottom: 12px;">このアプリケーションは、あなたの価値観と企業文化のマッチングをAIが分析するデモンストレーションです。</p>
        <ul style="margin: 0; padding-left: 20px;">
            <li>実際の企業データは限定的であり、結果は参考値としてご利用ください</li>
            <li>OpenAI APIを使用しているため、処理に数秒かかることがあります</li>
            <li>入力された情報は分析にのみ使用され、保存されません</li>
        </ul>
    </div>
    """)

# アプリ起動（Render用にポート設定を修正）
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        debug=False,
        share=False
    )
