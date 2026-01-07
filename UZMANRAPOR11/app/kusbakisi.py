# app/kusbakisi.py  (ISKO11 - hardcoded layout)
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import re, hashlib, colorsys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from PySide6.QtCore import Qt, QSize
from PySide6 import QtGui
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QGridLayout,
    QTableWidget, QTableWidgetItem, QComboBox, QPushButton, QFrame, QSizePolicy
)

from app import storage  # Usta Defteri sayımları + kısıt listeleri

IST = ZoneInfo("Europe/Istanbul")

# Daha dar salon arası boşluk
HALL_GAP_COLS = 1
LEFT_COLS = 6

# Bloklar: 15 / 18 / 18  -> aralara 1'er koridor satırı
TOP_BLOCK_ROWS = 15
MID_BLOCK_ROWS = 18
CORRIDOR_HEIGHT = 1  # boş satır adedi


def _seq(start: int, count: int, step: int):
    return [start + i * step for i in range(count)]


def _rows_spec_to_mapping(rows_spec, row_offset: int, col_offset: int):
    m = {}
    for r, (start, count, step) in enumerate(rows_spec):
        seq = _seq(start, count, step)
        for c, num in enumerate(seq):
            m[str(num)] = (row_offset + r, col_offset + c)
    return m


def _apply_row_corridors(mapping: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    """
    0..14   (15 satır)      -> aynı
    15..32  (18 satır)      -> +1 kaydır (15'te koridor var)
    33..50  (18 satır)      -> +2 kaydır (15 ve 34'te koridor var)
    """
    out: Dict[str, Tuple[int, int]] = {}
    cut1 = TOP_BLOCK_ROWS  # 15
    cut2 = TOP_BLOCK_ROWS + MID_BLOCK_ROWS  # 33

    for loom, (r, c) in mapping.items():
        rr = r
        if r >= cut1:
            rr += CORRIDOR_HEIGHT
        if r >= cut2:
            rr += CORRIDOR_HEIGHT
        out[loom] = (rr, c)
    return out


def _build_fixed_layout_from_spec() -> Dict[str, Tuple[int, int]]:
    mapping: Dict[str, Tuple[int, int]] = {}

    left_rows_spec = [
        (1822, 6, -1),
        (1806, 6, 2), (1805, 6, 2),
        (1804, 6, -2), (1803, 6, -2),
        (1782, 6, 2), (1781, 6, 2),
        (1780, 6, -2), (1779, 6, -2),
        (1758, 6, 2), (1757, 6, 2),
        (1756, 6, -2), (1755, 6, -2),
        (1734, 6, 2), (1733, 6, 2),
        (1301, 6, 2), (1302, 6, 2),
        (1323, 6, -2), (1324, 6, -2),
        (1325, 6, 2), (1326, 6, 2),
        (1347, 6, -2), (1348, 6, -2),
        (1349, 6, 2), (1350, 6, 2),
        (1371, 6, -2), (1372, 6, -2),
        (1373, 6, 2), (1374, 6, 2),
        (1395, 6, -2), (1396, 6, -2),
        (1397, 6, 2), (1398, 6, 2),
        (1517, 6, 2), (1518, 6, 2),
        (1539, 6, -2), (1540, 6, -2),
        (1541, 6, 2), (1542, 6, 2),
        (1563, 6, -2), (1564, 6, -2),
        (1565, 6, 2), (1566, 6, 2),
        (1587, 6, -2), (1588, 6, -2),
        (1589, 6, 2), (1590, 6, 2),
        (1611, 6, -2), (1612, 6, -2),
        (1613, 6, 2), (1614, 6, 2),
    ]

    right_rows_spec = [
        (1912, 6, -1),
        (1896, 6, 2), (1895, 6, 2),
        (1894, 6, -2), (1893, 6, -2),
        (1872, 6, 2), (1871, 6, 2),
        (1870, 6, -2), (1869, 6, -2),
        (1848, 6, 2), (1847, 6, 2),
        (1846, 6, -2), (1845, 6, -2),
        (1824, 6, 2), (1823, 6, 2),
        (1409, 6, 2), (1410, 6, 2),
        (1431, 6, -2), (1432, 6, -2),
        (1433, 6, 2), (1434, 6, 2),
        (1455, 6, -2), (1456, 6, -2),
        (1457, 6, 2), (1458, 6, 2),
        (1479, 6, -2), (1480, 6, -2),
        (1481, 6, 2), (1482, 6, 2),
        (1503, 6, -2), (1504, 6, -2),
        (1505, 6, 2), (1506, 6, 2),
        (1625, 6, 2), (1626, 6, 2),
        (1647, 6, -2), (1648, 6, -2),
        (1649, 6, 2), (1650, 6, 2),
        (1671, 6, -2), (1672, 6, -2),
        (1673, 6, 2), (1674, 6, 2),
        (1695, 6, -2), (1696, 6, -2),
        (1697, 6, 2), (1698, 6, 2),
        (1719, 6, -2), (1720, 6, -2),
        (1721, 6, 2), (1722, 6, 2),
    ]

    mapping.update(_rows_spec_to_mapping(left_rows_spec, row_offset=0, col_offset=0))
    right_offset = LEFT_COLS + HALL_GAP_COLS
    mapping.update(_rows_spec_to_mapping(right_rows_spec, row_offset=0, col_offset=right_offset))

    # kritik: koridor satırlarını uygula
    mapping = _apply_row_corridors(mapping)
    return mapping


MACHINE_LAYOUT = _build_fixed_layout_from_spec()

MAX_ROW = max((r for (r, c) in MACHINE_LAYOUT.values()), default=0)
MAX_COL = max((c for (r, c) in MACHINE_LAYOUT.values()), default=0)
TOTAL_ROWS = MAX_ROW + 1
TOTAL_COLS = MAX_COL + 1


def add_dividers_to_grid(grid):
    from PySide6.QtWidgets import QFrame

    v_div_col = LEFT_COLS  # gap başlangıcı

    # Dikey divider (sol-sağ arası)
    divider_v = QFrame()
    divider_v.setFrameShape(QFrame.VLine)
    divider_v.setStyleSheet("QFrame { background: #9aa0a6; }")
    divider_v.setFixedWidth(4)
    grid.addWidget(divider_v, 0, v_div_col, TOTAL_ROWS, HALL_GAP_COLS)

    # Yatay divider satırları (koridor satırları)
    h_div_rows = [
        TOP_BLOCK_ROWS,  # 15
        TOP_BLOCK_ROWS + CORRIDOR_HEIGHT + MID_BLOCK_ROWS,  # 34
    ]

    # Sol parça: 0..LEFT_COLS-1
    left_span = LEFT_COLS
    # Sağ parça: (LEFT_COLS + HALL_GAP_COLS) .. son
    right_start = LEFT_COLS + HALL_GAP_COLS
    right_span = TOTAL_COLS - right_start

    for cut_row in h_div_rows:
        # sol parça
        d1 = QFrame()
        d1.setFrameShape(QFrame.HLine)
        d1.setStyleSheet("QFrame { background: #9aa0a6; }")
        d1.setFixedHeight(4)
        grid.addWidget(d1, cut_row, 0, 1, left_span)

        # sağ parça
        d2 = QFrame()
        d2.setFrameShape(QFrame.HLine)
        d2.setStyleSheet("QFrame { background: #9aa0a6; }")
        d2.setFixedHeight(4)
        grid.addWidget(d2, cut_row, right_start, 1, right_span)


def _loom_in_category(loom_no: str, category: str) -> bool:
    # ISKO11’de kategori filtresi kullanmıyoruz (Tümü)
    return True


# -------------------------------------------------------------
#  Yardımcılar (normalize, renk, sıralama anahtarı)
# -------------------------------------------------------------

def _norm(s: object) -> str:
    return "" if s is None else str(s).strip()


def _hex_color_for_group(group_label: str) -> str:
    gl = _norm(group_label)
    if not gl:
        return "#eaeaea"
    h = hashlib.sha1(gl.encode("utf-8")).hexdigest()
    hue = int(h[:2], 16) * 360 // 255
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.52, 0.65)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _text_color_on(bg_hex: str) -> str:
    try:
        bg_hex = bg_hex.lstrip("#")
        r, g, b = int(bg_hex[0:2], 16), int(bg_hex[2:4], 16), int(bg_hex[4:6], 16)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "#000000" if lum > 155 else "#ffffff"
    except Exception:
        return "#000000"


_num_pat = re.compile(r"\d+(?:[.,]\d+)?")


def _fmt_num(f: float) -> str:
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    s = f"{f:.3f}".rstrip("0").rstrip(".")
    return s


def _normalize_tg_label(label: object) -> str:
    """
    '160,0 2 194,0' -> '160/2/194'
    '052.5/04/194'  -> '52.5/4/194'
    """
    s = _norm(label)
    if not s:
        return ""
    s = s.replace(",", ".")
    nums = [float(x.replace(",", ".")) for x in _num_pat.findall(s)]
    if not nums:
        return s
    parts = [_fmt_num(x) for x in nums]
    return "/".join(parts)


def _tarak_sort_key(label: str) -> tuple:
    s = _normalize_tg_label(label)
    nums = [float(x) for x in s.replace(",", ".").split("/") if x.strip() != ""]
    if not nums:
        return (9999.0,)
    return tuple(nums)


def _loom_digits(val: object) -> str:
    m = re.search(r"(\d+)", str(val or ""))
    return m.group(1) if m else ""


# -------------------------------------------------------------
#  DÜNÜN TOPLAM SAYIMI (3 vardiya toplamı)
# -------------------------------------------------------------

def _yesterday_shift_windows(ref: datetime | None = None):
    now = ref or datetime.now(IST)
    y = (now - timedelta(days=1)).date()
    s1 = datetime(y.year, y.month, y.day, 7, 0, tzinfo=IST)
    s2 = datetime(y.year, y.month, y.day, 15, 0, tzinfo=IST)
    s3 = datetime(y.year, y.month, y.day, 23, 0, tzinfo=IST)
    e1 = s2
    e2 = s3
    e3 = datetime(y.year, y.month, y.day, 7, 0, tzinfo=IST) + timedelta(days=1)
    return [(s1, e1), (s2, e2), (s3, e3)]


def _compute_yesterday_totals() -> tuple[int, int, str]:
    wins = _yesterday_shift_windows()
    total_dugum = 0
    total_takim = 0
    for (s, e) in wins:
        total_dugum += storage.count_usta_between(s, e, what="DÜĞÜM", direction="ALINDI")
        total_takim += storage.count_usta_between(s, e, what="TAKIM", direction="ALINDI")
    date_str = wins[0][0].strftime("%d.%m.%Y")
    return total_dugum, total_takim, date_str


# -------------------------------------------------------------
#  Görsel bileşenler
# -------------------------------------------------------------

@dataclass
class LoomView:
    loom: str
    tarak: str
    kalan_m: str
    is_empty: bool           # (Durus No 94 / open)
    is_running: bool         # running tablosunda var mı?
    color: str               # tarak grubu rengi (badge/şerit)
    koktip: str = ""
    cut_type: str = ""


class LoomCell(QLabel):
    def __init__(self, info: LoomView, white_bg: bool = False, parent=None):
        super().__init__(parent)

        # Kırmızı koşulları:
        # 1) Açık tezgah (Durus No 94 vb.) -> is_empty True
        # 2) Running'de değil -> is_running False
        loom_color = "#c00000" if (info.is_empty or (not info.is_running)) else "#000000"

        # Durum noktası (●)
        dot_html = (
            f"<span style='font-size:14pt;"
            f"color:{loom_color};"
            f"padding-right:6px;line-height:1;'>●</span>"
        )

        loom_badge = (
            f"<span style='display:inline-block;"
            f"font-size:12pt;font-weight:700;"
            f"background:#ffffff;color:{loom_color};"
            f"border:1px solid {loom_color};"
            f"border-radius:12px;padding:2px 10px;white-space:nowrap;'>"
            f"{dot_html}{info.loom}</span>"
        )

        cut_badge = ""
        if info.cut_type:
            cut_badge = (
                f"<span style='display:inline-block;"
                f"font-size:7pt;background:#ffffff;color:#111111;"
                f"border:1px solid #222222;border-radius:9px;"
                f"padding:1px 6px;white-space:nowrap;'>"
                f"{info.cut_type}</span>"
            )

        top_line = (
            f"<table width='100%' cellspacing='0' cellpadding='0'><tr>"
            f"<td align='left'>{loom_badge}</td>"
            f"<td align='right'>{cut_badge}</td>"
            f"</tr></table>"
        )

        t = info.tarak if info.tarak else "-"

        # Seçim filtresi varsa (white_bg=True) tarak şeridi gri olsun; yoksa tarak grubunun rengi
        if white_bg:
            strip_bg = "#eaeaea"
        else:
            strip_bg = info.color if (info.color and info.color != "#ffffff") else "#eaeaea"

        strip_fg = _text_color_on(strip_bg)

        # Tarak grubu: hücre içinde tam genişlik şerit
        middle_line = (
            f"<div style='margin-top:4px;width:100%;'>"
            f"  <div style='width:100%;"
            f"              background:{strip_bg};color:{strip_fg};"
            f"              font-size:10.5pt;font-weight:800;"
            f"              border:1px solid #666;border-radius:8px;"
            f"              padding:3px 6px;"
            f"              text-align:center;"
            f"              box-sizing:border-box;'>"
            f"{t}</div>"
            f"</div>"
        )

        km = info.kalan_m or ""
        kt = info.koktip or ""
        third_line_raw = (km if km else "") + ((" / " + kt) if (km and kt) else (kt if kt else ""))
        third_line = f"<div style='font-size:7pt; color:#222; margin-top:2px;'>{third_line_raw}</div>"

        self.setTextFormat(Qt.RichText)
        self.setText(f"{top_line}{middle_line}{third_line}")
        self.setAlignment(Qt.AlignCenter)
        self.setMargin(6)
        self.setWordWrap(True)

        border = "#b0b0b0"
        bg = "#ffffff"  # hücre her zaman beyaz
        self.setStyleSheet(f"""
            QLabel {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 6px;
                color: #000;
            }}
        """)

        self.setMinimumSize(QSize(96, 56))


# -------------------------------------------------------------
#  KUŞBAKIŞI WIDGET (ISKO11)
# -------------------------------------------------------------

class KusbakisiWidget(QWidget):
    """
    Sol: Özet tablo (Running’deki Tarak Grupları)
         + 'Takım olacak işler' (Running’de olmayan Dinamik gruplar)
    Sağ: Yerleşim ızgarası (hardcoded layout)
    Üstte KPI: Çalışan Tezgah / Dünün Alınan Düğüm & Takım
    """
    def __init__(self, parent=None):
        from PySide6.QtWidgets import QAbstractItemView
        super().__init__(parent)
        root = QHBoxLayout(self)

        # Sol panel
        self.sidebar = QWidget()
        left = QVBoxLayout(self.sidebar)

        self.sidebar.setMaximumWidth(515)
        self.sidebar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Kategori:"), 0)

        # ISKO11: kategori yok → sadece Tümü
        self.cmb_cat = QComboBox()
        self.cmb_cat.addItems(["Tümü"])

        self.lbl_working = QLabel()
        self.lbl_yesterday = QLabel()
        for w in (self.lbl_working, self.lbl_yesterday):
            w.setStyleSheet("QLabel { font-weight: 560; padding: 0 8px; }")
        self.lbl_yesterday.setTextFormat(Qt.RichText)
        self.lbl_yesterday.setWordWrap(True)

        self.btn_all_colors = QPushButton("Tümünü Renkli")

        toolbar.addWidget(self.cmb_cat, 0)
        toolbar.addSpacing(12)
        toolbar.addWidget(QLabel("Çalışan Tezgah:"), 0)
        toolbar.addWidget(self.lbl_working, 0)
        toolbar.addSpacing(8)
        toolbar.addWidget(self.lbl_yesterday, 0)
        self.lbl_yesterday.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        toolbar.addStretch(1)
        toolbar.addWidget(self.btn_all_colors, 0)

        self.lbl_status_kus = QLabel("")
        self.lbl_status_kus.setStyleSheet("QLabel{font-weight:700;}")
        left.addWidget(self.lbl_status_kus)
        left.addLayout(toolbar)

        self.tbl = QTableWidget(0, 6)
        self.tbl.setHorizontalHeaderLabels(
            ["Tarak Grubu", "İş Adedi", "Stok Adedi", "Tarak Adedi", "Açık Tezgah", "Termin (erken)"]
        )
        self.tbl.horizontalHeader().setStretchLastSection(True)
        left.addWidget(self.tbl, 8)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(QAbstractItemView.SingleSelection)

        left.addWidget(QLabel("Takım olacak işler"))
        self.tbl_planned = QTableWidget(0, 3)
        self.tbl_planned.setHorizontalHeaderLabels(["Tarak Grubu", "İş Adedi", "Stok Adedi"])
        self.tbl_planned.horizontalHeader().setStretchLastSection(True)
        left.addWidget(self.tbl_planned, 2)
        self.tbl_planned.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_planned.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_planned.setSelectionMode(QAbstractItemView.SingleSelection)

        root.addWidget(self.sidebar)

        # Sağ panel
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(6, 6, 6, 6)
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(6)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.grid_host)
        root.addWidget(self.scroll, 1)

        # Veri
        self.df_jobs: Optional[pd.DataFrame] = None
        self.df_run: Optional[pd.DataFrame] = None
        self.selected_group: Optional[str] = None

        self._kpi_working: int = 0

        # Kısıt listeleri
        self._blocked: set[str] = set()
        self._dummy: set[str] = set()

        # Sinyaller
        self.cmb_cat.currentTextChanged.connect(self._rebuild_all)
        self.tbl.cellClicked.connect(self._on_summary_clicked)
        self.btn_all_colors.clicked.connect(self._clear_selection)

    def _reload_restrictions(self):
        try:
            self._blocked = set(storage.load_blocked_looms() or [])
        except Exception:
            self._blocked = set()
        try:
            self._dummy = set(storage.load_dummy_looms() or [])
        except Exception:
            self._dummy = set()

    def _update_kpis(self):
        self.lbl_working.setText(str(self._kpi_working))
        try:
            tot_dugum, tot_takim, _ = _compute_yesterday_totals()
            self.lbl_yesterday.setText(
                f"<div><b>Alınan Düğüm :</b> {tot_dugum}<br>"
                f"<b>Alınan Takım :</b> {tot_takim}</div>"
            )
        except Exception:
            self.lbl_yesterday.setText("—")

    def refresh(self, df_jobs: Optional[pd.DataFrame], df_running: Optional[pd.DataFrame]) -> None:
        self._reload_restrictions()
        self.df_jobs = df_jobs.copy() if df_jobs is not None else None
        self.df_run = df_running.copy() if df_running is not None else None
        self._rebuild_all()

    def _rebuild_all(self) -> None:
        self._build_summary_tables()
        self._build_layout_grid()

    # ---------- Sol tablolar ----------
    def _build_summary_tables(self):
        self.tbl.setRowCount(0)
        self.tbl_planned.setRowCount(0)

        run_tg_set: set[str] = set()
        tarak_count = pd.Series(dtype=int)  # TOPLAM tarak (running + laststate)
        acik_count = pd.Series(dtype=int)  # ÇALIŞMAYAN tarak (running’de olmayanlar)
        working_count = 0

        # kısıtlı tezgahlar özetten tamamen çıkar (toplam sayımlardan da)
        ban = set(self._blocked) | set(self._dummy)

        # ----------------------------
        # 1) RUNNING tarafı
        # ----------------------------
        run = None
        run_lookup_looms: set[str] = set()
        running_count_by_tg = pd.Series(dtype=int)

        if self.df_run is not None and not self.df_run.empty:
            run = self.df_run.copy()

            run["_loom_digits"] = run.get("Tezgah No").astype(str).apply(_loom_digits)
            run = run[run["_loom_digits"].isin(set(MACHINE_LAYOUT.keys()))].copy()

            # kısıtlı tezgahlar çıkart
            run = run[~run["_loom_digits"].isin(ban)].copy()

            # KPI: çalışan tezgah (eski mantığın kalsın: Durus 94 çalışan sayılmasın)
            is_open = (run.get("Durus No", 0) == 94) | (run.get("_OpenTezgahFlag", False) == True)
            working_count = int((~is_open).sum())

            run["_tg"] = run.get("Tarak Grubu", "").apply(_normalize_tg_label)
            run_tg_set = set(run["_tg"].unique().tolist())

            run_lookup_looms = set(run["_loom_digits"].astype(str).tolist())

            # running adedi: durus 94 dahil (çünkü “çalışmayan” = running’de olmayanlar)
            running_count_by_tg = run.groupby("_tg", dropna=False).size()

        self._kpi_working = working_count
        self._update_kpis()

        # ----------------------------
        # 2) TOPLAM TARAK ADEDİ: Running + LastState
        # ----------------------------
        last_map = storage.load_loom_last_state_map()

        # loom -> tg evreni (ban hariç); öncelik running > laststate
        loom_to_tg: dict[str, str] = {}

        # running’den doldur
        if run is not None and not run.empty:
            for _, rr in run.iterrows():
                ld = str(rr.get("_loom_digits", "")).strip()
                if not ld:
                    continue
                tg = _normalize_tg_label(rr.get("Tarak Grubu", ""))
                if tg:
                    loom_to_tg[ld] = tg

        # laststate’den doldur (yalnız running’de olmayanlar)
        for loom_digits in MACHINE_LAYOUT.keys():
            if loom_digits in ban:
                continue
            if loom_digits in run_lookup_looms:
                continue
            ls = last_map.get(loom_digits)
            if not ls:
                continue
            tg = _normalize_tg_label(ls.get("TarakGrubu", ""))
            if tg:
                loom_to_tg[loom_digits] = tg

        if loom_to_tg:
            all_df = pd.DataFrame({"loom": list(loom_to_tg.keys()), "_tg": list(loom_to_tg.values())})
            tarak_count = all_df.groupby("_tg", dropna=False).size().rename("tarak_adedi")
        else:
            tarak_count = pd.Series(dtype=int, name="tarak_adedi")

        # Açık tezgah = çalışmayan = toplam - running
        # (running_count_by_tg yoksa 0 kabul edilir)
        running_aligned = running_count_by_tg.reindex(tarak_count.index).fillna(0).astype(int)
        acik_count = (tarak_count.astype(int) - running_aligned).astype(int).rename("acik")

        jobs = self.df_jobs
        if jobs is None or jobs.empty:
            return

        jobs = jobs.copy()
        jobs["_tg"] = jobs.get("Tarak Grubu", "").apply(_normalize_tg_label)
        if "_LeventHasDigits" in jobs.columns:
            jobs["_stok"] = jobs["_LeventHasDigits"].astype(bool)
        else:
            jobs["_stok"] = jobs.get("Levent No", "").astype(str).str.strip() != ""

        g_jobs = jobs.groupby("_tg", dropna=False)
        job_count = g_jobs.size().rename("is_adedi")
        stok_count = g_jobs["_stok"].sum(min_count=1).fillna(0).rename("stok_adedi")
        earliest_termin = g_jobs.apply(
            lambda x: pd.to_datetime(x.get("Mamul Termin"), errors="coerce").min()
        ).rename("termin")

        # 1) ÖZET TABLO (Running evreni)
        base = pd.DataFrame({"_tg": sorted(run_tg_set, key=_tarak_sort_key)})
        summary_main = base.merge(job_count.reset_index(), on="_tg", how="left") \
                           .merge(stok_count.reset_index(), on="_tg", how="left") \
                           .merge(earliest_termin.reset_index(), on="_tg", how="left") \
                           .merge(tarak_count.reset_index(), on="_tg", how="left") \
                           .merge(acik_count.reset_index(), on="_tg", how="left")

        summary_main[["is_adedi", "stok_adedi", "tarak_adedi", "acik"]] = summary_main[
            ["is_adedi", "stok_adedi", "tarak_adedi", "acik"]
        ].fillna(0)

        summary_main["termin"] = summary_main["termin"].apply(
            lambda d: "" if (pd.isna(d) or str(d) == "NaT") else pd.to_datetime(d).strftime("%d.%m.%Y")
        )
        summary_main["_sort"] = summary_main["_tg"].apply(_tarak_sort_key)
        summary_main = summary_main.sort_values(by="_sort", ascending=True)

        self.tbl.setRowCount(len(summary_main))
        for r, row in summary_main.iterrows():
            tg = _norm(row["_tg"])
            color = _hex_color_for_group(tg)
            fg = _text_color_on(color)
            vals = [
                tg,
                str(int(row["is_adedi"])),
                str(int(row["stok_adedi"])),
                str(int(row.get("tarak_adedi", 0))),
                str(int(row.get("acik", 0))),
                str(row["termin"]),
            ]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 0:
                    it.setBackground(QtGui.QColor(color))
                    it.setForeground(QtGui.QColor(fg))
                else:
                    it.setTextAlignment(Qt.AlignCenter)
                self.tbl.setItem(r, c, it)

        self.tbl.resizeColumnsToContents()
        if self.tbl.columnCount() > 0:
            self.tbl.setColumnWidth(0, max(self.tbl.columnWidth(0), 75))
            self.tbl.setColumnWidth(5, 80)

        # 2) TAKIM OLACAK İŞLER (Running’de olmayan Dinamik gruplar)
        extra_groups = [g for g in g_jobs.groups.keys() if g not in run_tg_set]
        summary_extra = pd.DataFrame({"_tg": sorted(extra_groups, key=_tarak_sort_key)})
        summary_extra = summary_extra.merge(job_count.reset_index(), on="_tg", how="left") \
                                     .merge(stok_count.reset_index(), on="_tg", how="left") \
                                     .merge(earliest_termin.reset_index(), on="_tg", how="left")
        summary_extra[["is_adedi", "stok_adedi"]] = summary_extra[["is_adedi", "stok_adedi"]].fillna(0)

        self.tbl_planned.setRowCount(len(summary_extra))
        for r, row in summary_extra.iterrows():
            tg = _norm(row["_tg"])
            vals = [tg, str(int(row["is_adedi"])), str(int(row["stok_adedi"]))]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c != 0:
                    it.setTextAlignment(Qt.AlignCenter)
                self.tbl_planned.setItem(r, c, it)

        self.tbl_planned.resizeColumnsToContents()
        if self.tbl_planned.columnCount() > 0:
            self.tbl_planned.setColumnWidth(0, max(self.tbl_planned.columnWidth(0), 120))

    # ---------- Sağ: yerleşim ızgarası ----------
    def _build_layout_grid(self):
        while self.grid.count():
            it = self.grid.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()

        run = self.df_run
        run_lookup = {}

        # --- Running varsa: tezgah->satır lookup üret ---
        if run is not None and not run.empty:
            run = run.copy()
            run["_loom_digits"] = run.get("Tezgah No", "").astype(str).apply(_loom_digits)

            # (Kategori filtresi kalsın; ISKO11'de True dönüyor)
            cat_sel = self.cmb_cat.currentText()
            mask = run.get("Tezgah No").astype(str).apply(lambda x: _loom_in_category(x, cat_sel))
            run = run[mask].copy()

            # lookup doldur
            for _, rr in run.iterrows():
                ld = str(rr.get("_loom_digits", "")).strip()
                if ld and ld not in run_lookup:
                    run_lookup[ld] = rr

        sel = _norm(self.selected_group) if self.selected_group else None
        sel_norm = _normalize_tg_label(sel) if sel else None

        blocked = set(self._blocked)
        dummy = set(self._dummy)
        last_map = storage.load_loom_last_state_map()

        # --- Layout'taki tüm tezgahları her zaman çiz ---
        for loom_digits, pos in MACHINE_LAYOUT.items():
            # Arızalı/Bakımda
            if loom_digits in blocked:
                self.grid.addWidget(
                    LoomCell(
                        LoomView(
                            loom=loom_digits,
                            tarak="Arızalı",
                            kalan_m="",
                            is_empty=False,
                            is_running=False,
                            color="#eaeaea",
                            koktip="",
                            cut_type=""
                        ),
                        white_bg=True
                    ),
                    pos[0], pos[1]
                )
                continue

            # Boş göster
            if loom_digits in dummy:
                self.grid.addWidget(
                    LoomCell(
                        LoomView(
                            loom=loom_digits,
                            tarak="Boş",
                            kalan_m="",
                            is_empty=False,
                            is_running=False,
                            color="#eaeaea",
                            koktip="",
                            cut_type=""
                        ),
                        white_bg=True
                    ),
                    pos[0], pos[1]
                )
                continue

            rr = run_lookup.get(loom_digits)
            if rr is None:
                ls = last_map.get(loom_digits)

                if ls:
                    tg_canon = _normalize_tg_label(ls.get("TarakGrubu", ""))
                    tip = (ls.get("DBirim", "") or "").strip()
                    color = _hex_color_for_group(tg_canon) if tg_canon else "#ffffff"

                    # seçim varsa ve bu tg seçili değilse dim
                    dim = (sel_norm is not None and tg_canon != sel_norm)

                    self.grid.addWidget(
                        LoomCell(
                            LoomView(
                                loom=loom_digits,
                                tarak=tg_canon or "-",
                                kalan_m="",
                                is_empty=False,
                                is_running=False,
                                color=color,
                                koktip=tip,
                                cut_type=""
                            ),
                            white_bg=dim
                        ),
                        pos[0], pos[1]
                    )
                else:
                    # Ne running ne laststate: tamamen boş
                    dim = (sel_norm is not None)
                    self.grid.addWidget(
                        LoomCell(
                            LoomView(
                                loom=loom_digits,
                                tarak="-",
                                kalan_m="",
                                is_empty=False,
                                is_running=False,
                                color="#ffffff",
                                koktip="",
                                cut_type=""
                            ),
                            white_bg=dim
                        ),
                        pos[0], pos[1]
                    )
                continue

            # --- Running varsa: dolu hücre ---
            tarak = _norm(rr.get("Tarak Grubu", ""))
            tarak_canon = _normalize_tg_label(tarak)
            color = _hex_color_for_group(tarak_canon)

            kalan = rr.get("_KalanMetreNorm", None)
            if pd.isna(kalan):
                kalan_s = ""
            else:
                try:
                    kalan_s = f"{float(kalan):.0f} m"
                except Exception:
                    kalan_s = str(kalan)

            koktip = str(
                rr.get("KökTip", "") or rr.get("Kök Tip Kodu", "") or rr.get("Tip No", "") or ""
            ).strip()

            cut_type = str(
                rr.get("ISAVER/ROTOCUT", "") or rr.get("Kesim Tipi", "") or ""
            ).strip()

            is_open = bool((rr.get("Durus No", 0) == 94) or (rr.get("_OpenTezgahFlag", False) == True))
            dim = (sel_norm is not None and tarak_canon != sel_norm)

            self.grid.addWidget(
                LoomCell(
                    LoomView(
                        loom=loom_digits,
                        tarak=tarak_canon,
                        kalan_m=kalan_s,
                        is_empty=is_open,
                        is_running=True,
                        color=color,
                        koktip=koktip,
                        cut_type=cut_type
                    ),
                    white_bg=dim
                ),
                pos[0], pos[1]
            )

        add_dividers_to_grid(self.grid)

    def _on_summary_clicked(self, row: int, col: int):
        item = self.tbl.item(row, 0)
        if not item:
            return
        self.selected_group = _norm(item.text()) or None
        self._build_layout_grid()

    def _clear_selection(self):
        self.selected_group = None
        self._build_layout_grid()

    def set_status_label(self, text: str, style: str | None = None):
        self.lbl_status_kus.setText(text or "")
        if style:
            self.lbl_status_kus.setStyleSheet(style)
