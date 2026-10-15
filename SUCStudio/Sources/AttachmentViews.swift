import SwiftUI
import AppKit
struct AttachmentLabel: View {
    let file:ChatAttachment
    var body: some View {
        HStack(spacing:8) {
            if let data=file.imageData,let image=NSImage(data:data) { Image(nsImage:image).resizable().scaledToFill().frame(width:38,height:38).clipped().clipShape(RoundedRectangle(cornerRadius:5)) }
            else { Image(systemName:"doc.text") }
            Text(file.name).lineLimit(1)
            if file.isImage { Text("immagine").font(StudioTypography.ui(.caption2)).foregroundStyle(.secondary) }
        }.font(StudioTypography.ui(.callout))
    }
}
