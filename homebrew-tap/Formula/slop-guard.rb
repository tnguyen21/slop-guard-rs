# This is a placeholder formula. It will be auto-updated by the release CI
# when a new version is tagged. See .github/workflows/release.yml for details.
#
# To use this tap:
#   brew tap tnguyen21/slop-guard
#   brew install slop-guard

class SlopGuard < Formula
  desc "Detect AI slop patterns in prose"
  homepage "https://github.com/tnguyen21/slop-guard-rs"
  version "0.1.0"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/tnguyen21/slop-guard-rs/releases/download/v0.1.0/slop-guard-macos-aarch64"
      sha256 "PLACEHOLDER_AARCH64_SHA256"
    else
      url "https://github.com/tnguyen21/slop-guard-rs/releases/download/v0.1.0/slop-guard-macos-x86_64"
      sha256 "PLACEHOLDER_X86_64_SHA256"
    end
  end

  on_linux do
    url "https://github.com/tnguyen21/slop-guard-rs/releases/download/v0.1.0/slop-guard-linux-x86_64-musl"
    sha256 "PLACEHOLDER_LINUX_MUSL_SHA256"
  end

  def install
    bin.install stable.url.split("/").last => "slop-guard"
  end

  test do
    assert_match "slop-guard", shell_output("#{bin}/slop-guard --version")
  end
end
