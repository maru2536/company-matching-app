# -*- coding: utf-8 -*-
"""app.py - グラデーションモダンデザインの企業文化マッチングアプリ (Render用に最適化)"""

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

# 友人の会社のサイトURL
FRIEND_BASE_URL = "https://friend-company.co.jp/result/receive"

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
    if not OPENAI_API_KEY:
        return """
        1. 自由度重視ワークスタイルタイプ
           自分の裁量で働き方を決められる環境を好みます。成果が出ていれば、どのように仕事を進めるかは自由に選びたいと考えています。
        
        2. 成長志向アチーバータイプ
           企業の急速な成長とともに自己のスキルアップも目指しています。変化を楽しむことができ、常に進化を求める環境で活躍したいと思っています。
        
        3. 独立自由プロフェッショナルタイプ
           経済的な自立を重視し、場所を選ばずに働ける自由なスタイルを望んでいます。生活のためだけでなく、自己実現を目指す働き方を求めています。
        """
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
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
        
        res = client.chat.completions.create(
            model=model, 
            messages=[{"role":"user","content":prompt}],
            temperature=0.7
        )
        return res.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in classify_values: {str(e)}")
        return f"""
        1. 自由度重視ワークスタイルタイプ
           自分の裁量で働き方を決められる環境を好みます。成果が出ていれば、どのように仕事を進めるかは自由に選びたいと考えています。
        
        2. 成長志向アチーバータイプ
           企業の急速な成長とともに自己のスキルアップも目指しています。変化を楽しむことができ、常に進化を求める環境で活躍したいと思っています。
        
        3. 独立自由プロフェッショナルタイプ
           経済的な自立を重視し、場所を選ばずに働ける自由なスタイルを望んでいます。生活のためだけでなく、自己実現を目指す働き方を求めています。
        """

def compute_top3(summary, df):
    """企業とのマッチング計算"""
    try:
        # APIキーがない場合はフォールバックデータを返す
        if not OPENAI_API_KEY:
            return [
                {"会社名": "テックイノベーション株式会社", "初期": 72.3, "中期": 78.5, "最近": 85.7, 
                 "文化特性": {"革新性": 85, "安定性": 70, "成長機会": 90, "環境": 75, "報酬": 80}},
                {"会社名": "グローバルソリューションズ", "初期": 68.1, "中期": 72.4, "最近": 79.8,
                 "文化特性": {"革新性": 75, "安定性": 85, "成長機会": 70, "環境": 65, "報酬": 90}},
                {"会社名": "フューチャークリエイト", "初期": 65.7, "中期": 70.2, "最近": 76.4,
                 "文化特性": {"革新性": 90, "安定性": 60, "成長機会": 85, "環境": 80, "報酬": 75}}
            ]
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # ユーザーベクトル取得
        user_vecs = []
        for line in summary.splitlines():
            if not line.strip():
                continue
            try:
                r = client.embeddings.create(input=line, model="text-embedding-ada-002")
                user_vecs.append(np.array(r.data[0].embedding))
            except:
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
        
        return sorted(results, key=lambda x: x["最近"] or 0, reverse=True)[:3]
    except Exception as e:
        logger.error(f"Error in compute_top3: {str(e)}")
        
        # エラー時のフォールバックデータ
        return [
            {"会社名": "テックイノベーション株式会社", "初期": 72.3, "中期": 78.5, "最近": 85.7, 
             "文化特性": {"革新性": 85, "安定性": 70, "成長機会": 90, "環境": 75, "報酬": 80}},
            {"会社名": "グローバルソリューションズ", "初期": 68.1, "中期": 72.4, "最近": 79.8,
             "文化特性": {"革新性": 75, "安定性": 85, "成長機会": 70, "環境": 65, "報酬": 90}},
            {"会社名": "フューチャークリエイト", "初期": 65.7, "中期": 70.2, "最近": 76.4,
             "文化特性": {"革新性": 90, "安定性": 60, "成長機会": 85, "環境": 80, "報酬": 75}}
        ]

def generate_match_reason(company_data, user_values, model="gpt-3.5-turbo"):
    """マッチング理由を生成"""
    if not OPENAI_API_KEY:
        # 企業ごとのフォールバックメッセージ
        if "テック" in company_data["会社名"]:
            return "あなたの革新性を重視する価値観が、同社の先進的な企業文化と高くマッチしています。特に成長機会の豊富さと挑戦を奨励する環境があなたの可能性を最大限に引き出すでしょう。"
        elif "グローバル" in company_data["会社名"]:
            return "あなたの安定志向と報酬重視の価値観が、同社の堅実な企業文化と高い親和性を持っています。特に充実した福利厚生と明確なキャリアパスがあなたの長期的な成長をサポートします。"
        else:
            return "あなたの創造性と自由度を重視する価値観が、同社の柔軟な企業文化と強く共鳴しています。特にチームの多様性と協働の環境があなたの独自のアイデアを形にする機会を提供するでしょう。"
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
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
        
        res = client.chat.completions.create(
            model=model, 
            messages=[{"role":"user","content":prompt}],
            max_tokens=200
        )
        
        return res.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in generate_match_reason: {str(e)}")
        
        # 企業ごとのフォールバックメッセージ
        if "テック" in company_data["会社名"]:
            return "あなたの革新性を重視する価値観が、同社の先進的な企業文化と高くマッチしています。特に成長機会の豊富さと挑戦を奨励する環境があなたの可能性を最大限に引き出すでしょう。"
        elif "グローバル" in company_data["会社名"]:
            return "あなたの安定志向と報酬重視の価値観が、同社の堅実な企業文化と高い親和性を持っています。特に充実した福利厚生と明確なキャリアパスがあなたの長期的な成長をサポートします。"
        else:
            return "あなたの創造性と自由度を重視する価値観が、同社の柔軟な企業文化と強く共鳴しています。特にチームの多様性と協働の環境があなたの独自のアイデアを形にする機会を提供するでしょう。"

def run_app(q1_choice, q1_text, q2_choice, q2_text, q3_choice, q3_text,
            q4_choice, q4_text, q5_choice, q5_text):
    """メイン分析処理"""
    try:
        # 最初に処理中メッセージを表示
        progress_html = """
        <div style="text-align: center; padding: 30px 0;">
            <div style="display: inline-block; width: 40px; height: 40px; border: 4px solid rgba(255,255,255,0.3); 
                 border-top-color: #7C3AED; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <p style="margin-top: 15px; font-weight: bold; color: #7C3AED;">分析を実行中です...</p>
            <style>@keyframes spin { to { transform: rotate(360deg); } }</style>
        </div>
        """
        
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
            return "回答が入力されていません", "", gr.update(visible=False), gr.update(value="次へ進む", variant="primary", elem_classes="")
        
        # 価値観分類
        summary = classify_values(", ".join(answers))
        
        # 企業マッチング計算
        top3 = compute_top3(summary, df)
        
        # 各企業のマッチング理由を生成
        for company in top3:
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
        <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 40px 20px; background: linear-gradient(135deg, #f5f7fa, #f8f9fa); border-radius: 20px; text-align: center;">
            <div style="background: white; border-radius: 16px; padding: 40px; box-shadow: 0 12px 40px rgba(0,0,0,0.1); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.3);">
                <div style="font-size: 48px; margin-bottom: 20px;">🎉</div>
                <h2 style="background: linear-gradient(to right, #4F46E5, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 28px; margin-bottom: 20px; font-weight: bold;">分析が完了しました！</h2>
                
                <p style="font-size: 18px; color: #64748b; margin-bottom: 30px; line-height: 1.6;">
                    あなたの価値観に基づいた企業マッチング結果をご用意いたしました。<br>
                    詳細な結果は、専用サイトでご確認いただけます。
                </p>
                
                <div style="background: linear-gradient(135deg, rgba(79, 70, 229, 0.05), rgba(236, 72, 153, 0.05)); border-radius: 12px; padding: 20px; margin-bottom: 30px; border: 1px solid rgba(124, 58, 237, 0.2);">
                    <div style="font-size: 16px; color: #475569; margin-bottom: 15px;">
                        ✅ 価値観タイプの分類<br>
                        ✅ おすすめ企業TOP3<br>
                        ✅ マッチング理由の詳細分析<br>
                        ✅ 企業文化特性の比較
                    </div>
                </div>
                
                <a href="{friend_url}" target="_blank" style="display: inline-block; background: linear-gradient(to right, #4F46E5, #7C3AED); color: white; text-decoration: none; padding: 16px 32px; border-radius: 50px; font-size: 18px; font-weight: bold; box-shadow: 0 8px 25px rgba(124, 58, 237, 0.3); transition: all 0.3s ease; transform: translateY(0);">
                    📊 分析結果を確認する
                </a>
                
                <p style="font-size: 14px; color: #94a3b8; margin-top: 20px;">
                    ※ リンク先で簡単な登録を行うと、詳細な分析結果をご覧いただけます
                </p>
            </div>
        </div>
        
        <style>
        a:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 35px rgba(124, 58, 237, 0.4) !important;
            filter: brightness(1.05) !important;
        }}
        </style>
        """
        
        # 最終結果を返す（リンクのみ表示）
        return summary, result_link_html, gr.update(visible=True), gr.update(value="分析完了 ✓", variant="secondary", elem_classes="")
        
    except Exception as e:
        logger.error(f"Error in run_app: {str(e)}")
        error_html = f"""
        <div style="text-align: center; padding: 30px; background: #fef2f2; border-radius: 16px; box-shadow: 0 8px 30px rgba(0,0,0,0.05);">
            <h3 style="color: #be123c; font-size: 20px; margin-bottom: 15px;">エラーが発生しました</h3>
            <p>申し訳ありませんが、処理中にエラーが発生しました。</p>
            <p style="font-family: monospace; background: rgba(0,0,0,0.03); padding: 10px; border-radius: 8px; margin: 15px 0; font-size: 14px;">{str(e)[:200]}</p>
            <p>再度お試しいただくか、システム管理者にお問い合わせください。</p>
        </div>
        """
        return f"エラーが発生しました: {str(e)}", error_html, gr.update(visible=True), gr.update(value="再試行", variant="primary", elem_classes="")

# CSS定義（グラデーションモダンスタイル）
custom_css = """
body {
  font-family: 'Segoe UI', Arial, sans-serif;
  color: #333;
  background: linear-gradient(135deg, #f5f7fa, #f8f9fa);
}

/* カードスタイル */
.content-card {
  background: white;
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 20px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.2);
  transition: all 0.3s ease;
}

.content-card:hover {
  box-shadow: 0 12px 40px rgba(0,0,0,0.08);
  transform: translateY(-2px);
}

/* 質問タイトル */
.content-card h3 {
  background: linear-gradient(to right, #4F46E5, #7C3AED);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-size: 20px;
  margin-bottom: 15px;
}

/* ボタンスタイル */
.primary-button {
  background: linear-gradient(to right, #4F46E5, #7C3AED) !important;
  color: white !important;
  transition: all 0.3s ease !important;
  transform: translateY(0) !important;
  box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3) !important;
  position: relative !important;
  border-radius: 100px !important;
  padding: 2px 0 !important;
}

.primary-button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 16px rgba(124, 58, 237, 0.4) !important;
  filter: brightness(1.05) !important;
}

/* ヘッダー */
h1 {
  background: linear-gradient(to right, #4F46E5, #EC4899);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: bold;
  text-align: center;
  margin: 30px 0 !important;
  font-size: 32px !important;
  padding-bottom: 5px;
}

/* モバイル対応 */
@media (max-width: 768px) {
  .content-card {
    padding: 20px;
  }
}

@keyframes spin {
  to {transform: rotate(360deg);}
}
"""

# UI定義
with gr.Blocks(css=custom_css, title="企業文化マッチング") as demo:
    gr.HTML("<h1>企業文化マッチング</h1>")
    
    with gr.Group(elem_classes="content-card"):
        gr.HTML("<h3>Q1: 仕事で大切にしたいこと</h3>")
        q1_choice = gr.CheckboxGroup(
            ["挑戦", "安定性", "社会貢献性", "自己成長", "柔軟な働き方", "プライベートの充実"],
            label="選択（複数選択可）"
        )
        q1_text = gr.Textbox(label="その他", placeholder="その他の価値観を入力してください...")

    with gr.Group(elem_classes="content-card"):
        gr.HTML("<h3>Q2: どんな働き方がしたいか</h3>")
        q2_choice = gr.CheckboxGroup(
            ["フルリモートワーク", "出社", "リモートワークと出社のハイブリッド"],
            label="選択（複数選択可）"
        )
        q2_text = gr.Textbox(label="その他", placeholder="具体的な働き方があれば入力してください...")

    with gr.Group(elem_classes="content-card"):
        gr.HTML("<h3>Q3: どんなチームで働きたいか</h3>")
        q3_choice = gr.CheckboxGroup(
            ["裁量権大", "多様性", "強いリーダーシップ", "フラットな関係性"],
            label="選択（複数選択可）"
        )
        q3_text = gr.Textbox(label="その他", placeholder="理想のチーム環境について入力してください...")

    with gr.Group(elem_classes="content-card"):
        gr.HTML("<h3>Q4: どんな環境を求めるか</h3>")
        q4_choice = gr.CheckboxGroup(
            ["研修が充実している", "OJTがある", "海外研修がある", "自己学習の支援がある", "副業OK"],
            label="選択（複数選択可）"
        )
        q4_text = gr.Textbox(label="その他", placeholder="その他の環境要件を入力してください...")

    with gr.Group(elem_classes="content-card"):
        gr.HTML("<h3>Q5: その他重視するポイント</h3>")
        q5_choice = gr.CheckboxGroup(
            ["高インセンティブ", "勤務地", "フレックス", "スピード感"],
            label="選択（複数選択可）"
        )
        q5_text = gr.Textbox(label="その他", placeholder="その他に重視する点があれば入力してください...")

    with gr.Row(elem_classes="button-row"):
        next_btn = gr.Button("分析を開始する", elem_classes="primary-button", size="lg")

    with gr.Group(visible=False) as results_area:
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
with gr.Accordion("使用上の注意", open=False, elem_classes="footer"):
    gr.HTML("""
    <div style="padding: 10px;">
        <p><strong>このアプリについて:</strong></p>
        <ul>
            <li>このアプリは、あなたの価値観と企業文化のマッチングをシミュレーションするデモです。</li>
            <li>実際の企業データは限定的であり、結果はあくまで参考値としてご利用ください。</li>
            <li>OpenAI APIを使用しているため、処理に数秒かかることがあります。</li>
            <li>問題が発生した場合は、ページをリロードするか、後ほど再度お試しください。</li>
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
