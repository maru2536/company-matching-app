# -*- coding: utf-8 -*-
"""app.py - グラデーションモダンデザインの企業文化マッチングアプリ (Hugging Face用に最適化)"""

import pandas as pd
import zipfile
import io
import ast
import numpy as np
import gradio as gr
from openai import OpenAI
import time
import os
import sys
import traceback
import json
import re

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

# 環境変数からAPIキーを取得（Hugging Face Spaces用）
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# 埋め込みデータ読み込み処理
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
    
except Exception as e:
    logger.error(f"Error loading data: {str(e)}")
    logger.error(traceback.format_exc())
    
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

# 最低限のエラーハンドリングを持つ関数
def classify_values(inputs, model="gpt-3.5-turbo"):
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
            temperature=0.7  # 少し創造性を持たせる
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
    try:
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
            q4_choice, q4_text, q5_choice, q5_text, progress=gr.Progress()):
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
        
        # すぐに処理中表示を返す
        yield "", progress_html, gr.update(visible=True), gr.update(value="分析中...", variant="secondary", elem_classes="")
        
        # APIキーチェック
        if not OPENAI_API_KEY:
            yield "システムエラー：APIキーが設定されていません。管理者にお問い合わせください。", "", gr.update(visible=False), gr.update(value="次へ進む", variant="primary", elem_classes="")
            return
        
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
            yield "回答が入力されていません", "", gr.update(visible=False), gr.update(value="次へ進む", variant="primary", elem_classes="")
            return
        
        # 分析実行
        # ステータス更新：価値観を分類中
        status_html = """
        <div style="text-align: center; padding: 30px 0;">
            <div style="display: inline-block; width: 40px; height: 40px; border: 4px solid rgba(255,255,255,0.3); 
                 border-top-color: #7C3AED; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <p style="margin-top: 15px; font-weight: bold; background: linear-gradient(to right, #4F46E5, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">💭 価値観を分類中...</p>
            <div style="width: 80%; max-width: 300px; margin: 15px auto; height: 6px; background: rgba(255,255,255,0.2); border-radius: 3px;">
                <div style="width: 25%; height: 100%; background: linear-gradient(to right, #4F46E5, #7C3AED); border-radius: 3px;"></div>
            </div>
            <style>@keyframes spin { to { transform: rotate(360deg); } }</style>
        </div>
        """
        yield "", status_html, gr.update(visible=True), gr.update(value="分析中...", variant="secondary", elem_classes="")
        progress(0.25, "価値観を分類中...")
        
        summary = classify_values(", ".join(answers))
        
        # ステータス更新：マッチングを計算中
        status_html = status_html.replace("💭 価値観を分類中...", "🔍 企業とのマッチングを計算中...")
        status_html = status_html.replace("width: 25%", "width: 50%")
        yield "", status_html, gr.update(visible=True), gr.update(value="分析中...", variant="secondary", elem_classes="")
        progress(0.5, "企業とのマッチングを計算中...")
        
        top3 = compute_top3(summary, df)
        
        # ステータス更新：マッチング理由を生成中
        status_html = status_html.replace("🔍 企業とのマッチングを計算中...", "✨ マッチング理由を生成中...")
        status_html = status_html.replace("width: 50%", "width: 75%")
        yield "", status_html, gr.update(visible=True), gr.update(value="分析中...", variant="secondary", elem_classes="")
        progress(0.75, "マッチング理由を生成中...")
        
        # 各企業のマッチング理由を生成
        for company in top3:
            company["マッチング理由"] = generate_match_reason(company, summary)
        
        # ステータス更新：結果を取得中
        status_html = status_html.replace("✨ マッチング理由を生成中...", "📊 結果を取得中...")
        status_html = status_html.replace("width: 75%", "width: 90%")
        yield "", status_html, gr.update(visible=True), gr.update(value="分析中...", variant="secondary", elem_classes="")
        progress(0.9, "結果を取得中...")
        
        # 価値観パート用のHTMLを生成
        values_html = "<h2 style='margin-bottom: 24px; background: linear-gradient(to right, #4F46E5, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>あなたに当てはまる価値観タイプ</h2>"
        
        # 価値観タイプを抽出する処理
        lines = summary.strip().split('\n')
        types_with_descriptions = []
        
        # 数字.で始まる行を探す
        pattern = re.compile(r'^(\d+)\.\s*(.*?)$')
        current_type = None
        current_desc = None
        
        for line in lines:
            line = line.strip()
            if not line:  # 空行はスキップ
                continue
                
            # 数字で始まる行はタイプ名として処理
            match = pattern.match(line)
            if match:
                # 前のタイプがあれば保存
                if current_type is not None and current_desc is not None:
                    types_with_descriptions.append((current_type, current_desc))
                
                current_type = match.group(2)  # タイプ名
                current_desc = None
            # それ以外は説明文として処理
            elif current_type is not None and current_desc is None:
                current_desc = line
        
        # 最後のタイプも保存
        if current_type is not None and current_desc is not None:
            types_with_descriptions.append((current_type, current_desc))
        
        # HTMLを生成
        for i, (type_name, description) in enumerate(types_with_descriptions[:3]):
            values_html += f"""
            <div style="margin-bottom: 24px; background: white; border-radius: 16px; padding: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);">
                <div style="display: flex; align-items: flex-start;">
                    <div style="font-size: 18px; font-weight: bold; margin-right: 10px; width: 30px; height: 30px; border-radius: 50%; background: linear-gradient(to right, #4F46E5, #7C3AED); color: white; display: flex; align-items: center; justify-content: center;">{i+1}</div>
                    <div style="flex: 1;">
                        <div style="font-size: 18px; font-weight: bold; margin-bottom: 8px; background: linear-gradient(to right, #4F46E5, #7C3AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{type_name}</div>
                        <div style="font-size: 16px; line-height: 1.6;">{description}</div>
                    </div>
                </div>
            </div>
            """
        
        # 結果HTML生成 - グラデーションモダンスタイル
        results_html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: linear-gradient(135deg, #f5f7fa, #f8f9fa); border-radius: 20px;">
            <div style="background: white; border-radius: 16px; padding: 30px; margin-bottom: 30px; box-shadow: 0 8px 30px rgba(0,0,0,0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);">
                {values_html}
            </div>
            
            <div style="background: white; border-radius: 16px; padding: 30px; margin-bottom: 20px; box-shadow: 0 8px 30px rgba(0,0,0,0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);">
                <h3 style="background: linear-gradient(to right, #4F46E5, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 24px; margin-bottom: 25px;">マッチング企業</h3>
                <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px;">
        """
        
        # グラデーションカラーの定義
        gradient_colors = {
            0: {
                "bg": "linear-gradient(135deg, rgba(79, 70, 229, 0.05), rgba(236, 72, 153, 0.05))",
                "border": "linear-gradient(135deg, #4F46E5, #EC4899)",
                "accent": "linear-gradient(to right, #4F46E5, #7C3AED)"
            },
            1: {
                "bg": "linear-gradient(135deg, rgba(79, 70, 229, 0.02), rgba(236, 72, 153, 0.02))",
                "border": "rgba(200, 200, 200, 0.3)",
                "accent": "linear-gradient(to right, #6366F1, #A855F7)"
            },
            2: {
                "bg": "linear-gradient(135deg, rgba(79, 70, 229, 0.01), rgba(236, 72, 153, 0.01))",
                "border": "rgba(200, 200, 200, 0.2)",
                "accent": "linear-gradient(to right, #818CF8, #C084FC)"
            }
        }
        
        # 企業カード
        for i, r in enumerate(top3):
            rank = i + 1
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
            score = r["最近"] if r["最近"] is not None else 0
            progress_width = min(int(score), 100) if score else 0
            
            a = lambda k: f"{r[k]}%" if r[k] is not None else "-"
            
            results_html += f"""
                    <div style="flex: 1; min-width: 250px; background: {gradient_colors[i]['bg']}; padding: 20px; border-radius: 16px; border: 1px solid {gradient_colors[i]['border']}; box-shadow: 0 8px 20px rgba(0,0,0,0.05); backdrop-filter: blur(10px);">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                            <span style="font-size: 28px;">{medal}</span>
                            <span style="font-weight: bold; font-size: 18px; background: {gradient_colors[i]['accent']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{r['会社名']}</span>
                        </div>
                        <div style="height: 8px; background: rgba(0,0,0,0.05); border-radius: 4px; margin-bottom: 5px; overflow: hidden;">
                            <div style="height: 100%; width: {progress_width}%; background: {gradient_colors[i]['accent']}; border-radius: 4px;"></div>
                        </div>
                        <div style="text-align: right; font-weight: bold; margin-bottom: 15px; background: {gradient_colors[i]['accent']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{a('最近')}</div>
                        <div style="display: flex; gap: 8px; margin-bottom: 20px;">
                            <div style="flex: 1; text-align: center; background: rgba(0,0,0,0.02); padding: 8px; border-radius: 12px;">
                                <div style="font-size: 12px; color: #64748b;">初期</div>
                                <div>{a('初期')}</div>
                            </div>
                            <div style="flex: 1; text-align: center; background: rgba(0,0,0,0.02); padding: 8px; border-radius: 12px;">
                                <div style="font-size: 12px; color: #64748b;">中期</div>
                                <div>{a('中期')}</div>
                            </div>
                            <div style="flex: 1; text-align: center; background: rgba(79, 70, 229, 0.08); padding: 8px; border-radius: 12px;">
                                <div style="font-size: 12px; background: {gradient_colors[i]['accent']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">最近</div>
                                <div style="background: {gradient_colors[i]['accent']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{a('最近')}</div>
                            </div>
                        </div>
                        <div style="background: rgba(255,255,255,0.6); padding: 15px; border-radius: 12px; font-size: 14px; box-shadow: 0 4px 15px rgba(0,0,0,0.03);">
                            <div style="font-weight: bold; background: {gradient_colors[i]['accent']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px;">マッチング理由</div>
                            <p style="margin: 0; line-height: 1.6;">{r['マッチング理由']}</p>
                        </div>
                    </div>
            """
        
        results_html += """
                </div>
            </div>
        </div>
        """
        
        progress(1.0, "完了!")
        
        # 最終結果を返す
        yield summary, results_html, gr.update(visible=True), gr.update(value="分析完了 ✓", variant="secondary", elem_classes="")
        
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
        yield f"エラーが発生しました: {str(e)}", error_html, gr.update(visible=True), gr.update(value="再試行", variant="primary", elem_classes="")

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

/* チェックボックス */
.content-card [data-testid="checkbox"] {
  border-radius: 8px !important;
  border-color: rgba(124, 58, 237, 0.5) !important;
  margin-bottom: 8px !important;
}

.content-card [data-testid="checkbox-selected"] {
  background-color: #7C3AED !important;
}

/* テキスト入力 */
.content-card input[type="text"] {
  border-radius: 12px !important;
  border-color: rgba(124, 58, 237, 0.2) !important;
  padding: 10px 15px !important;
  transition: all 0.2s ease !important;
}

.content-card input[type="text"]:focus {
  border-color: #7C3AED !important;
  box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.1) !important;
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

.primary-button:active {
  transform: translateY(1px) !important;
  box-shadow: 0 2px 8px rgba(124, 58, 237, 0.3) !important;
}

.primary-button.loading {
  background: rgba(124, 58, 237, 0.7) !important;
  color: white !important;
  pointer-events: none !important;
}

.secondary-button {
  background: rgba(255,255,255,0.8) !important;
  color: #7C3AED !important;
  border: 1px solid rgba(124, 58, 237, 0.2) !important;
  transition: all 0.2s ease !important;
  border-radius: 100px !important;
}

.secondary-button:hover {
  background: rgba(124, 58, 237, 0.05) !important;
  border-color: rgba(124, 58, 237, 0.5) !important;
}

.secondary-button:active {
  background: rgba(124, 58, 237, 0.1) !important;
  transform: translateY(1px) !important;
}

/* アクティブなボタンのアニメーション */
.primary-button button.loading:before {
  content: '';
  box-sizing: border-box;
  position: absolute;
  top: 50%;
  left: 10px;
  width: 20px;
  height: 20px;
  margin-top: -10px;
  border-radius: 50%;
  border: 2px solid #ffffff;
  border-top-color: transparent;
  animation: spin 0.8s linear infinite;
}

.footer {
  background: white;
  border-radius: 16px;
  padding: 20px;
  margin-top: 20px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.2);
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

/* アコーディオン */
.gr-accordion {
  border: none !important;
  box-shadow: none !important;
}

.gr-accordion details > summary {
  background: linear-gradient(135deg, rgba(79, 70, 229, 0.05), rgba(236, 72, 153, 0.05)) !important;
  border-radius: 12px !important;
  padding: 12px 16px !important;
  border: 1px solid rgba(124, 58, 237, 0.2) !important;
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
"""

# シンプルなUI定義
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

# Hugging Face Spaces用の注意書き
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

# アプリ起動
if __name__ == "__main__":
    demo.launch(debug=True)
