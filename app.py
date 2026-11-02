# -*- coding: utf-8 -*-
"""
반도체 공정 장비 이상 감지 모니터링
2026 도전학기(2) — 김재민 · 장주왕 / 지도교수 이진태
데이터: UCI SECOM (McCann & Johnston, 2008)
"""

import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

DATA_URL = ("https://raw.githubusercontent.com/sharmaroshan/"
            "SECOM-Detecting-Defected-Items/master/uci-secom.csv")
목표재현율 = 0.85

st.set_page_config(page_title="공정 이상 감지 모니터링",
                   page_icon="📶", layout="wide")

# ────────────────────────────── 스타일 ──────────────────────────────
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1280px;}
  [data-testid="stMetricValue"] {font-size: 1.75rem;}
  [data-testid="stMetricLabel"] {font-size: 0.82rem; opacity: 0.75;}
  .caption-sm {font-size: 0.8rem; opacity: 0.65; line-height: 1.5;}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────── 데이터·모델 ──────────────────────────────
@st.cache_data(show_spinner="공정 데이터를 불러오는 중…")
def 데이터준비():
    """02번 노트북과 완전히 같은 전처리. 분할을 먼저 하고 대치·표준화는 학습용에만 맞춘다."""
    df = pd.read_csv(DATA_URL)
    X = df.drop(columns=["Time", "Pass/Fail"])
    y = (df["Pass/Fail"] == 1).astype(int)

    X = X.drop(columns=X.columns[X.nunique() <= 1])
    X = X.drop(columns=X.columns[X.isna().mean() > 0.5])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    시험원본 = X_te.copy()

    imputer = SimpleImputer(strategy="median").fit(X_tr)
    scaler = StandardScaler().fit(imputer.transform(X_tr))

    def 변환(d):
        return pd.DataFrame(scaler.transform(imputer.transform(d)), columns=X.columns)

    return {
        "columns": list(X.columns),
        "X_tr": 변환(X_tr), "y_tr": y_tr.reset_index(drop=True),
        "X_te": 변환(X_te), "y_te": y_te.reset_index(drop=True),
        "시험원본": 시험원본.reset_index(drop=True),
        "imputer": imputer, "scaler": scaler,
        "전체표본": len(df), "전체불량": int(y.sum()),
    }


@st.cache_resource(show_spinner="모델을 학습하는 중… (최초 1회, 약 30초)")
def 모델준비(_D):
    """랜덤 포레스트 학습 + 교차검증으로 임계값 자동 산출 (05번 노트북과 동일한 규칙)."""
    X_tr, y_tr = _D["X_tr"], _D["y_tr"]
    모델 = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=42, n_jobs=-1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv확률 = cross_val_predict(모델, X_tr, y_tr, cv=cv,
                             method="predict_proba", n_jobs=-1)[:, 1]

    pr, rc, th = precision_recall_curve(y_tr, cv확률)
    pr, rc = pr[:-1], rc[:-1]
    가능 = rc >= 목표재현율
    기본임계값 = float(th[np.where(가능)[0][np.argmax(pr[가능])]]) if 가능.any() else 0.5

    모델.fit(X_tr, y_tr)
    중요도 = pd.Series(모델.feature_importances_, index=X_tr.columns)

    return {"model": 모델, "threshold": 기본임계값,
            "cv확률": cv확률, "top20": 중요도.sort_values(ascending=False).head(20)}


D = 데이터준비()
M = 모델준비(D)


# ────────────────────────────── 사이드바 ──────────────────────────────
with st.sidebar:
    st.subheader("설정")

    데이터원 = st.radio(
        "검사할 데이터",
        ["시험 데이터 (314장)", "CSV 업로드"],
        help="시험 데이터는 모델이 학습에 한 번도 쓰지 않은 웨이퍼입니다.")

    업로드 = None
    if 데이터원 == "CSV 업로드":
        업로드 = st.file_uploader(
            "센서값 CSV", type="csv",
            help=f"열 이름이 센서 번호인 표. 누락된 센서는 중앙값으로 채웁니다. "
                 f"(기대 센서 {len(D['columns'])}개)")

    st.divider()

    st.markdown("**판정 임계값**")
    권장임계값 = round(M["threshold"], 4)
    임계값 = st.slider(
        "위험도가 이 값 이상이면 알람", 0.02, 0.50,
        value=권장임계값, step=0.0025, format="%.4f",
        label_visibility="collapsed")

    if abs(임계값 - 권장임계값) < 1e-9:
        st.caption(f"현재 값은 교차검증으로 자동 산출된 권장값입니다 "
                   f"(재현율 {목표재현율} 이상을 만족하는 값 중 정밀도 최대).")
    else:
        방향 = "낮추면 더 많이 잡지만 검사 물량이 늘어납니다." if 임계값 < 권장임계값 \
            else "높이면 검사 물량은 줄지만 놓치는 불량이 늘어납니다."
        st.caption(f"권장값 {권장임계값:.4f} 에서 벗어났습니다. {방향}")

    st.divider()
    st.caption(
        f"**모델** 랜덤 포레스트 (나무 300그루)  \n"
        f"**학습** {len(D['y_tr'])}장 · 불량 {int(D['y_tr'].sum())}장  \n"
        f"**원본** SECOM {D['전체표본']}장 · 불량 {D['전체불량']}장 "
        f"({D['전체불량']/D['전체표본']*100:.2f}%)")


# ────────────────────────────── 검사 실행 ──────────────────────────────
st.title("반도체 공정 장비 이상 감지")
st.markdown(
    '<p class="caption-sm">센서값으로 웨이퍼의 불량 위험도를 매기고, 임계값을 넘는 건에 '
    '알람을 올립니다. 2026 도전학기(2) · 지도교수 이진태</p>',
    unsafe_allow_html=True)

라벨 = None
if 데이터원 == "CSV 업로드":
    if 업로드 is None:
        st.info("왼쪽에서 CSV를 올리거나 시험 데이터를 선택하세요.")
        st.stop()
    원본 = pd.read_csv(io.BytesIO(업로드.getvalue()))
    if "Pass/Fail" in 원본.columns:
        라벨 = (원본["Pass/Fail"] == 1).astype(int).reset_index(drop=True)
    원본 = 원본.reindex(columns=D["columns"])
    빠진 = 원본.columns[원본.isna().all()]
    if len(빠진):
        st.warning(f"CSV에 없는 센서 {len(빠진)}개는 학습 데이터의 중앙값으로 채웠습니다.")
    X = pd.DataFrame(D["scaler"].transform(D["imputer"].transform(원본)),
                     columns=D["columns"])
    표시원본 = 원본.reset_index(drop=True)
else:
    X = D["X_te"]
    라벨 = D["y_te"]
    표시원본 = D["시험원본"]

위험도 = M["model"].predict_proba(X)[:, 1]
알람 = 위험도 >= 임계값

# ────────────────────────────── 요약 지표 ──────────────────────────────
if 라벨 is not None:
    tn, fp, fn, tp = confusion_matrix(라벨, 알람.astype(int), labels=[0, 1]).ravel()
    c = st.columns(5)
    c[0].metric("검사 대상", f"{len(X):,}장")
    c[1].metric("🔴 이상 판정", f"{int(알람.sum()):,}건",
                f"{알람.mean()*100:.1f}% 재검사")
    c[2].metric("잡은 불량", f"{tp}/{tp+fn}장",
                f"재현율 {tp/(tp+fn):.3f}" if tp + fn else "불량 없음")
    c[3].metric("놓친 불량", f"{fn}장",
                delta=f"-{fn}장" if fn else "없음",
                delta_color="inverse" if fn else "off")
    c[4].metric("정밀도", f"{tp/(tp+fp):.3f}" if tp + fp else "—",
                f"이상 판정 {tp+fp}건 중 {tp}건 적중")

    if tp + fn:
        if tp / (tp + fn) >= 목표재현율:
            st.success(f"목표 재현율 {목표재현율} 달성 — 실제 불량 {tp+fn}장 중 {tp}장을 "
                       f"잡았습니다 (재현율 {tp/(tp+fn):.3f}). "
                       f"대가로 정상 {fp}장이 함께 재검사 대상이 됩니다.")
        else:
            st.error(f"목표 재현율 {목표재현율} 미달 — 재현율 {tp/(tp+fn):.3f}. "
                     f"왼쪽에서 임계값을 낮추세요.")
else:
    c = st.columns(3)
    c[0].metric("검사 대상", f"{len(X):,}장")
    c[1].metric("🔴 이상 판정", f"{int(알람.sum()):,}건", f"{알람.mean()*100:.1f}% 재검사")
    c[2].metric("최고 위험도", f"{위험도.max():.3f}")
    st.info("정답 라벨이 없는 데이터입니다. 재현율·정밀도는 계산할 수 없습니다.")

st.divider()

# ────────────────────────────── 위험도 분포 ──────────────────────────────
왼, 오 = st.columns([3, 2], gap="large")

with 왼:
    st.subheader("위험도 분포")
    분포 = pd.DataFrame({
        "위험도": 위험도,
        "판정": np.where(알람, "이상", "정상"),
    })
    if 라벨 is not None:
        분포["실제"] = np.where(라벨 == 1, "불량", "정상")
        색 = alt.Color("실제:N",
                       scale=alt.Scale(domain=["정상", "불량"],
                                       range=["#9DB4C8", "#E45756"]),
                       legend=alt.Legend(title=None, orient="top-right"))
    else:
        색 = alt.Color("판정:N",
                       scale=alt.Scale(domain=["정상", "이상"],
                                       range=["#9DB4C8", "#E45756"]),
                       legend=alt.Legend(title=None, orient="top-right"))

    막대 = alt.Chart(분포).mark_bar().encode(
        x=alt.X("위험도:Q", bin=alt.Bin(maxbins=45), title="위험도"),
        y=alt.Y("count():Q", title="웨이퍼 수",
                axis=alt.Axis(titleAngle=0, titleAlign="left", titleX=-30, titleY=-12),
                stack=True),
        color=색,
        tooltip=[alt.Tooltip("count():Q", title="웨이퍼 수")],
    )
    선 = alt.Chart(pd.DataFrame({"x": [임계값]})).mark_rule(
        color="#333", strokeDash=[6, 4], size=1.6).encode(x="x:Q")
    글 = alt.Chart(pd.DataFrame({"x": [임계값], "t": [f"임계값 {임계값:.4f}"]})).mark_text(
        align="left", dx=6, dy=-6, baseline="top", fontSize=11,
        color="#333").encode(x="x:Q", y=alt.value(0), text="t:N")

    st.altair_chart((막대 + 선 + 글).properties(height=300),
                    use_container_width=True)
    st.markdown(
        '<p class="caption-sm">점선 오른쪽이 이상 판정 대상입니다. 불량(빨강)이 '
        '오른쪽에 치우쳐 있을수록 모델이 위험 순위를 잘 매기고 있다는 뜻입니다.</p>',
        unsafe_allow_html=True)

with 오:
    st.subheader("임계값을 바꾸면")
    격자 = np.linspace(0.02, 0.50, 120)
    행 = []
    for t in 격자:
        p = 위험도 >= t
        if 라벨 is not None and 라벨.sum():
            행.append({"임계값": t, "값": float((p & (라벨 == 1)).sum() / 라벨.sum()),
                       "지표": "재현율"})
        행.append({"임계값": t, "값": float(p.mean()), "지표": "검사 비율"})
    곡선 = pd.DataFrame(행)

    기준 = alt.Chart(pd.DataFrame({"y": [목표재현율]})).mark_rule(
        color="#BBB", strokeDash=[4, 4]).encode(y="y:Q")
    현재 = alt.Chart(pd.DataFrame({"x": [임계값]})).mark_rule(
        color="#333", strokeDash=[6, 4], size=1.6).encode(x="x:Q")
    라인 = alt.Chart(곡선).mark_line(size=2.2).encode(
        x=alt.X("임계값:Q", title="임계값"),
        y=alt.Y("값:Q", title="비율",
                axis=alt.Axis(titleAngle=0, titleAlign="left", titleX=-26, titleY=-12),
                scale=alt.Scale(domain=[0, 1.02])),
        color=alt.Color("지표:N",
                        scale=alt.Scale(domain=["재현율", "검사 비율"],
                                        range=["#E45756", "#54A24B"]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["지표:N", alt.Tooltip("임계값:Q", format=".3f"),
                 alt.Tooltip("값:Q", format=".3f")],
    )
    st.altair_chart((라인 + 기준 + 현재).properties(height=300),
                    use_container_width=True)
    st.markdown(
        '<p class="caption-sm">회색 가로선이 목표 재현율 0.85입니다. 두 선의 간격이 '
        '곧 "불량 하나를 더 잡기 위해 검사해야 하는 추가 물량"입니다.</p>',
        unsafe_allow_html=True)

st.divider()

# ────────────────────────────── 알람 목록 ──────────────────────────────
st.subheader("위험도 순위")

목록 = pd.DataFrame({
    "웨이퍼": [f"W-{i:04d}" for i in range(len(X))],
    "위험도": 위험도.round(4),
    "판정": np.where(알람, "🔴 이상", "🟢 정상"),
})
if 라벨 is not None:
    목록["실제"] = np.where(라벨 == 1, "불량", "정상")
    목록["결과"] = np.select(
        [알람 & (라벨 == 1), 알람 & (라벨 == 0), (~알람) & (라벨 == 1)],
        ["적중", "헛알람", "놓침"],
        default="정상 통과")

목록 = 목록.sort_values("위험도", ascending=False).reset_index(drop=True)

보기 = st.radio("보기", ["이상 판정만", "전체"], horizontal=True,
                label_visibility="collapsed")
표 = 목록[목록["판정"] == "🔴 이상"] if 보기 == "이상 판정만" else 목록

st.dataframe(
    표, use_container_width=True, height=330, hide_index=True,
    column_config={
        "위험도": st.column_config.ProgressColumn(
            "위험도", min_value=0.0,
            max_value=float(max(위험도.max(), 임계값 * 1.5)), format="%.4f"),
    })
st.download_button("이 목록 CSV로 받기", 표.to_csv(index=False).encode("utf-8-sig"),
                   file_name="알람목록.csv", mime="text/csv")

st.divider()

# ────────────────────────────── 센서 들여다보기 ──────────────────────────────
좌, 우 = st.columns([2, 3], gap="large")

with 좌:
    st.subheader("모델이 보는 센서")
    중요 = M["top20"].reset_index()
    중요.columns = ["센서", "중요도"]
    중요["센서"] = "센서 " + 중요["센서"].astype(str)
    st.altair_chart(
        alt.Chart(중요).mark_bar(color="#4C78A8").encode(
            x=alt.X("중요도:Q", title="중요도"),
            y=alt.Y("센서:N", sort="-x", title=None,
                    axis=alt.Axis(labelOverlap=False, labelFontSize=11)),
            tooltip=["센서:N", alt.Tooltip("중요도:Q", format=".4f")],
        ).properties(height=520),
        use_container_width=True)
    st.markdown(
        '<p class="caption-sm">446개 센서 중 모델이 판단에 가장 많이 참고한 20개입니다. '
        '현장 점검은 이 센서부터 보는 것이 효율적입니다.</p>',
        unsafe_allow_html=True)

with 우:
    st.subheader("웨이퍼 한 장 들여다보기")
    선택 = st.selectbox("웨이퍼 선택", 목록["웨이퍼"].tolist(),
                       help="위험도 높은 순서입니다.")
    idx = int(선택.split("-")[1])

    행 = 목록[목록["웨이퍼"] == 선택].iloc[0]

    # 계기판 신호등: 정상(초록) / 이상(빨강)
    이상여부 = 행["판정"] == "🔴 이상"
    신호색, 신호배경, 신호글 = (("#C62828", "#FDECEA", "이상 — 재검사 대상")
                          if 이상여부 else ("#1E8E3E", "#E8F5E9", "정상 — 통과"))
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:16px;background:{신호배경};
            border:2px solid {신호색};border-radius:14px;padding:14px 18px;margin:6px 0 14px 0">
  <div style="width:46px;height:46px;border-radius:50%;background:{신호색};
              box-shadow:0 0 14px {신호색}88;flex-shrink:0"></div>
  <div>
    <div style="font-size:1.45rem;font-weight:700;color:{신호색};line-height:1.2">{신호글}</div>
    <div style="font-size:0.85rem;color:#444">위험도 {행['위험도']:.4f} · 임계값 {임계값:.4f}</div>
  </div>
</div>""", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("위험도", f"{행['위험도']:.4f}")
    m2.metric("판정", 행["판정"])
    m3.metric("실제", 행.get("실제", "—"))

    센서들 = [c for c in M["top20"].index]
    상세 = pd.DataFrame({
        "센서": ["센서 " + str(c) for c in 센서들],
        "표준화값": [float(X.iloc[idx][c]) for c in 센서들],
        "원본값": [float(표시원본.iloc[idx][c])
                 if c in 표시원본.columns and pd.notna(표시원본.iloc[idx][c]) else np.nan
                 for c in 센서들],
    })
    상세["이탈"] = 상세["표준화값"].abs().round(2)

    st.altair_chart(
        alt.Chart(상세).mark_bar().encode(
            x=alt.X("표준화값:Q", title="평균에서 벗어난 정도 (표준편차 단위)"),
            y=alt.Y("센서:N", sort=상세["센서"].tolist(), title=None,
                    axis=alt.Axis(labelOverlap=False, labelFontSize=11)),
            color=alt.condition(alt.datum.표준화값 > 0,
                                alt.value("#E45756"), alt.value("#4C78A8")),
            tooltip=["센서:N", alt.Tooltip("표준화값:Q", format=".2f"),
                     alt.Tooltip("원본값:Q", format=".4f")],
        ).properties(height=520),
        use_container_width=True)
    st.markdown(
        '<p class="caption-sm">0이 학습 데이터의 평균입니다. 막대가 길수록 그 센서가 '
        '평소와 다른 값을 보였다는 뜻입니다 (빨강 = 평균보다 높음).</p>',
        unsafe_allow_html=True)

# ────────────────────────────── 모델 성능 ──────────────────────────────
st.divider()
with st.expander("모델 성능과 한계 — 읽고 쓰세요", expanded=False):
    st.markdown(f"""
**학습과 평가 방법**

- 전체 {D['전체표본']}장을 8:2로 층화 분할. 시험용 {len(D['y_te'])}장은 모델 학습과
  임계값 선정에 **한 번도 쓰지 않았습니다**.
- 임계값은 학습 데이터 {len(D['y_tr'])}장 안에서 5겹 교차검증으로만 정했습니다
  (권장값 {M['threshold']:.4f}).
- 비교한 감지 방법 5종: 로지스틱 회귀 · 랜덤 포레스트 · XGBoost ·
  Isolation Forest · One-Class SVM. PR-AUC가 가장 높은 랜덤 포레스트를 썼습니다.

**이 화면이 말하지 않는 것**

- 정밀도가 0.1 수준입니다. 알람 10건 중 9건은 정상품입니다. 이 화면은
  "불량을 놓치지 않는 것"에 맞춰져 있고, 그 대가로 재검사 물량이 늘어납니다.
  재검사 비용이 불량 유출 비용보다 큰 공정이라면 임계값을 올려야 합니다.
- SECOM 데이터는 센서 번호만 공개되어 있고 각 센서가 공정의 어느 지점을 재는지는
  밝혀져 있지 않습니다. 따라서 "몇 번 센서가 중요하다"까지는 말할 수 있어도
  "어느 설비를 점검하라"까지는 이 데이터만으로 말할 수 없습니다.
- 정확도(Accuracy)는 표시하지 않았습니다. 불량률이 6.6%라 전부 "정상"이라고만
  답해도 93.4%가 나오는 지표이기 때문입니다.

**출처** McCann, M. & Johnston, A. (2008). *SECOM* [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C54305
    """)

st.markdown(
    '<p class="caption-sm">2026 도전학기(2) · 반도체 공정 데이터를 활용한 장비 이상 감지 '
    '프로그램 개발 및 모니터링 화면 구현 · 김재민 · 장주왕 / 지도교수 이진태</p>',
    unsafe_allow_html=True)
