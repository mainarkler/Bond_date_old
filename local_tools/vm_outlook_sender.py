"""
Локальная автоматизация VM PDF -> классический Outlook.

Установка:
    py -m pip install playwright pywin32
    py -m playwright install msedge

Конфигурация хранится вне GitHub:
    %USERPROFILE%\\vm_report_sender.json

Пример:
{
  "url": "https://bonddate.streamlit.app/?action=vm_pdf",
  "recipients": [
    "коллега1@company.ru",
    "коллега2@company.ru"
  ],
  "subject": "VM отчет GDZ6",
  "send": false
}

Сначала оставьте send=false: письмо будет создано в Outlook, но не отправлено.
После проверки поставьте send=true.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import win32com.client
from playwright.sync_api import sync_playwright


CONFIG_PATH = Path(os.environ["USERPROFILE"]) / "vm_report_sender.json"
DOWNLOAD_DIR = Path(os.environ["TEMP"]) / "vm_report_sender"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Не найден файл конфигурации: {CONFIG_PATH}\n"
            "Создайте его по примеру в комментарии в начале скрипта."
        )

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    recipients = config.get("recipients", [])
    if isinstance(recipients, str):
        recipients = [recipients]
    recipients = [str(x).strip() for x in recipients if str(x).strip()]

    if not recipients:
        raise ValueError("В конфигурации нет recipients.")

    return {
        "url": str(config.get("url", "https://bonddate.streamlit.app/?action=vm_pdf")).strip(),
        "recipients": recipients,
        "subject": str(
            config.get(
                "subject",
                f"VM отчет {datetime.now():%d.%m.%Y}",
            )
        ).strip(),
        "send": bool(config.get("send", False)),
        "timeout_ms": int(config.get("timeout_ms", 180000)),
    }


def download_pdf(url: str, timeout_ms: int) -> Path:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Удаляем старые PDF, чтобы случайно не отправить вчерашний файл.
    for old_file in DOWNLOAD_DIR.glob("*.pdf"):
        try:
            old_file.unlink()
        except OSError:
            pass

    with sync_playwright() as p:
        # Используем установленный Microsoft Edge, если он доступен.
        browser = p.chromium.launch(
            channel="msedge",
            headless=True,
        )
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

        # Streamlit может сначала показывать загрузку данных, поэтому ждём кнопку.
        button = page.get_by_role("button", name="Скачать VM PDF")
        button.wait_for(state="visible", timeout=timeout_ms)

        with page.expect_download(timeout=timeout_ms) as download_info:
            button.click()

        download = download_info.value
        filename = download.suggested_filename or f"VM_{datetime.now():%Y-%m-%d}.pdf"
        target = DOWNLOAD_DIR / filename
        download.save_as(str(target))

        context.close()
        browser.close()

    if not target.exists() or target.stat().st_size < 10_000:
        raise RuntimeError(f"PDF скачан некорректно: {target}")

    return target


def create_outlook_message(pdf_path: Path, recipients: list[str], subject: str, send: bool) -> None:
    outlook = win32com.client.Dispatch("Outlook.Application")
    mail = outlook.CreateItem(0)  # olMailItem

    mail.To = "; ".join(recipients)
    mail.Subject = subject

    # Пользователь просил не дублировать данные PDF в теле письма.
    mail.Body = ""
    mail.Attachments.Add(str(pdf_path))

    if send:
        mail.Send()
        print(f"Письмо отправлено: {', '.join(recipients)}")
    else:
        mail.Display()
        print("Письмо создано в Outlook, но НЕ отправлено (send=false).")


def main() -> int:
    try:
        config = load_config()
        print(f"Получаю PDF: {config['url']}")
        pdf_path = download_pdf(config["url"], config["timeout_ms"])
        print(f"PDF получен: {pdf_path}")

        create_outlook_message(
            pdf_path=pdf_path,
            recipients=config["recipients"],
            subject=config["subject"],
            send=config["send"],
        )
        return 0
    except Exception as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
