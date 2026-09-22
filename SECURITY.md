# Security / 安全说明

This application is intended for local use and has no production authentication layer. Keep its loopback binding. Cloud generation sends selected evidence and optional images to the configured provider. Review documents before using a cloud backend.

应用面向本地使用，未实现生产级身份认证，请保留本机监听。云端生成会发送所选证据和可选图片，使用前确认文档适合交给该服务处理。

Keep secrets in an ignored `.env` or environment variables. Do not attach original private documents, keys or unsanitized logs to issues. Report reproducible non-sensitive problems through GitHub issues; use private vulnerability reporting if enabled for security-sensitive details.

密钥放在被忽略的 `.env` 或环境变量中。Issue 不附私人原件、密钥或未经脱敏的日志；安全敏感细节使用仓库已启用时的私密漏洞报告功能。
