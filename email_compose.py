import re
from datetime import datetime
from email.message import EmailMessage
from urllib.parse import quote, urlencode

import streamlit as st


def parse_email_list(raw_recipients: str) -> tuple[list[str], list[str]]:
    chunks = re.split(r"[;,\s]+", raw_recipients.strip()) if raw_recipients else []
    unique = []
    seen = set()
    email_pattern = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
    invalid = []
    for item in chunks:
        email = item.strip()
        if not email:
            continue
        if email in seen:
            continue
        seen.add(email)
        if email_pattern.match(email):
            unique.append(email)
        else:
            invalid.append(email)
    return unique, invalid


def build_compose_link(service: str, recipients: list[str], cc_recipients: list[str], subject: str, body: str) -> str:
    def query_string(params: dict[str, str]) -> str:
        parts = []
        for key, value in params.items():
            if value:
                parts.append((key, value))
        return urlencode(parts, quote_via=quote)

    to_field = ";".join(recipients)
    cc_field = ";".join(cc_recipients)
    if service == "Почтовый клиент по умолчанию":
        mailto_params = query_string({"cc": cc_field, "subject": subject, "body": body})
        return f"mailto:{to_field}" + (f"?{mailto_params}" if mailto_params else "")
    if service == "Gmail":
        return "https://mail.google.com/mail/?" + query_string(
            {"view": "cm", "fs": "1", "to": to_field, "cc": cc_field, "su": subject, "body": body}
        )
    if service == "Outlook Web":
        return "https://outlook.office.com/mail/deeplink/compose?" + query_string(
            {"to": to_field, "cc": cc_field, "subject": subject, "body": body}
        )
    if service == "Yandex Mail":
        return "https://mail.yandex.ru/compose?" + query_string(
            {"to": to_field, "cc": cc_field, "subject": subject, "body": body}
        )
    return "https://e.mail.ru/compose/?" + query_string(
        {"To": to_field, "Cc": cc_field, "Subject": subject, "Body": body}
    )


def build_eml_attachment(
    recipients: list[str],
    cc_recipients: list[str],
    subject: str,
    body: str,
    attachment_name: str | None = None,
    attachment_bytes: bytes | None = None,
    extra_attachments: list[tuple[str, bytes, str, str]] | None = None,
) -> bytes:
    message = EmailMessage()
    message["To"] = ", ".join(recipients)
    if cc_recipients:
        message["Cc"] = ", ".join(cc_recipients)
    message["Subject"] = subject
    message.set_content(body)
    if attachment_name and attachment_bytes:
        message.add_attachment(
            attachment_bytes,
            maintype="application",
            subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=attachment_name,
        )
    for extra_name, extra_bytes, maintype, subtype in extra_attachments or []:
        message.add_attachment(
            extra_bytes,
            maintype=maintype,
            subtype=subtype,
            filename=extra_name,
        )
    return message.as_bytes()


def render_email_compose_section(
    report_title: str,
    key_prefix: str,
    attachment_name: str | None = None,
    attachment_bytes: bytes | None = None,
    default_body: str | None = None,
    extra_attachments: list[tuple[str, bytes, str, str]] | None = None,
):
    st.markdown("---")
    if st.button("📧 Подготовить письмо", key=f"{key_prefix}_open_compose"):
        st.session_state[f"{key_prefix}_show_compose"] = True

    if not st.session_state.get(f"{key_prefix}_show_compose", False):
        return

    st.subheader("📧 Отправка отчёта по почте")
    st.caption(
        "Выберите сервис и адреса — приложение откроет черновик письма. "
        "Можно добавить адреса в копию и скачать .eml с Excel-вложением. "
        "Подпись/бланк подставляется почтовым клиентом по вашим настройкам."
    )

    mail_service = st.selectbox(
        "Почтовый сервис",
        ["Почтовый клиент по умолчанию", "Gmail", "Outlook Web", "Yandex Mail", "Mail.ru"],
        key=f"{key_prefix}_service",
    )
    recipients_raw = st.text_area(
        "Адреса получателей (через запятую, точку с запятой или перенос строки)",
        placeholder="user1@example.com; user2@example.com",
        key=f"{key_prefix}_recipients",
    )
    cc_recipients_raw = st.text_area(
        "Адреса в копии (CC)",
        placeholder="copy1@example.com; copy2@example.com",
        key=f"{key_prefix}_cc_recipients",
    )
    default_subject = f"{report_title} на {datetime.today().strftime('%d.%m.%Y')}"
    mail_subject = st.text_input("Тема письма", value=default_subject, key=f"{key_prefix}_subject")
    body_value = st.session_state.get(f"{key_prefix}_default_body") or (
        "Коллеги, добрый день!\n\n"
        f"Направляю {report_title.lower()}.\n"
        "Пожалуйста, см. вложенный файл.\n\n"
    )
    mail_body = st.text_area(
        "Текст письма",
        value=body_value,
        height=180,
        key=f"{key_prefix}_body",
    )

    if st.button("Сгенерировать письмо", key=f"{key_prefix}_generate"):
        recipients, invalid_recipients = parse_email_list(recipients_raw)
        cc_recipients, invalid_cc_recipients = parse_email_list(cc_recipients_raw)
        invalid_emails = invalid_recipients + invalid_cc_recipients
        if invalid_emails:
            st.error(
                "Некорректные адреса: "
                + ", ".join(invalid_emails[:10])
                + ("..." if len(invalid_emails) > 10 else "")
            )
        if not recipients:
            st.warning("Укажите хотя бы один корректный email получателя.")
        if recipients:
            subject = mail_subject.strip()
            body = mail_body.strip()
            compose_link = build_compose_link(mail_service, recipients, cc_recipients, subject, body)
            st.success(f"Черновик подготовлен для {len(recipients)} получателя(ей).")
            st.link_button("Открыть письмо в выбранном сервисе", compose_link)
            st.code(compose_link, language="text")
            if attachment_name and attachment_bytes:
                eml_bytes = build_eml_attachment(
                    recipients,
                    cc_recipients,
                    subject,
                    body,
                    attachment_name,
                    attachment_bytes,
                    extra_attachments=extra_attachments,
                )
                st.download_button(
                    "📎 Скачать черновик .eml с Excel-вложением",
                    data=eml_bytes,
                    file_name="report_with_attachment.eml",
                    mime="message/rfc822",
                    key=f"{key_prefix}_eml_download",
                )
