import MLXToolKit
import XCTest
@testable import MLXQwen25VL

/// The 3B weights are Qwen-Research-licensed (non-commercial), not Apache-2.0 (AB-A-0142).
final class LicenseTests: XCTestCase {

    func testTheWeightsAreDeclaredQwenResearch() {
        let license = Qwen25VLPackage.manifest.license
        XCTAssertEqual(license.weightLicense.identifier, "LicenseRef-Qwen-Research")
        XCTAssertEqual(license.portCodeLicense, .apache2, "the port code stays Apache-2.0")
        XCTAssertEqual(Qwen25VLPackage.manifest.provenance.sourceRepo, "mlx-community/Qwen2.5-VL-3B-Instruct-bf16")
    }

    func testAPermissiveOnlyPolicyNeverAdmitsIt() {
        let license = Qwen25VLPackage.manifest.license
        XCTAssertFalse(license.weightLicense.isPermissive)
        XCTAssertFalse(license.weightLicense.isEvalAcknowledged)
        XCTAssertEqual(LicensePolicy.permissiveOnly.evaluate(license), .rejectedWeight(.qwenResearch))
        XCTAssertEqual(LicensePolicy.permissiveOrAcknowledged.evaluate(license), .rejectedWeight(.qwenResearch))
        XCTAssertEqual(LicensePolicy.any.evaluate(license), .admitted, "named-only hosts can still load it")
    }

    func testEveryConfigurationLoadsA3BRepo() {
        // The declaration covers what the package loads: both quant tiers are 3B conversions.
        for quant in [Quant.bf16, .int4] {
            XCTAssertTrue(Qwen25VLConfiguration(quant: quant).repo.contains("Qwen2.5-VL-3B-Instruct"), "\(quant)")
        }
    }
}
