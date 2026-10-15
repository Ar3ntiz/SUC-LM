import AppKit

// Use exactly the SF Symbol used by the idle menu-bar item, at every icon size.
let output = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
guard let symbol = NSImage(systemSymbolName: "sparkles.rectangle.stack", accessibilityDescription: nil)?
    .withSymbolConfiguration(NSImage.SymbolConfiguration(pointSize: 700, weight: .regular))?
    .withSymbolConfiguration(NSImage.SymbolConfiguration(paletteColors: [.white])) else {
    fatalError("SF Symbol sparkles.rectangle.stack unavailable")
}
func render(_ pixels: Int) throws -> Data {
    let image = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    image.size = NSSize(width: 1024, height: 1024)
    NSGraphicsContext.saveGraphicsState()
    let graphics = NSGraphicsContext(bitmapImageRep: image)!
    NSGraphicsContext.current = graphics
    graphics.imageInterpolation = .high
    let tile = NSBezierPath(roundedRect: NSRect(x: 56, y: 56, width: 912, height: 912), xRadius: 205, yRadius: 205)
    let gradient = NSGradient(starting: NSColor(srgbRed: 0.15, green: 0.17, blue: 0.39, alpha: 1),
        ending: NSColor(srgbRed: 0.32, green: 0.39, blue: 0.78, alpha: 1))!
    gradient.draw(in: tile, angle: 65)
    // Fit the original glyph without changing its proportions or stroke shape.
    let aspect = symbol.size.width / symbol.size.height
    let width: CGFloat = 670, height: CGFloat = 670 / aspect
    symbol.draw(in: NSRect(x: (1024-width)/2, y: (1024-height)/2, width: width, height: height),
        from: .zero, operation: .sourceOver, fraction: 1)
    NSGraphicsContext.restoreGraphicsState()
    return image.representation(using: .png, properties: [:])!
}
for size in [16, 32, 128, 256, 512] {
    try render(size).write(to: output.appendingPathComponent("icon_\(size)x\(size).png"))
    try render(size*2).write(to: output.appendingPathComponent("icon_\(size)x\(size)@2x.png"))
}
