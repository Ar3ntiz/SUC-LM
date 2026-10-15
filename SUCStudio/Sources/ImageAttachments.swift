import Foundation
import ImageIO
import UniformTypeIdentifiers

extension AttachmentReader {
    static let maxImageBytes = 2 * 1024 * 1024
    static let maxTotalImageBytes = 8 * 1024 * 1024
    static func readImage(_ data:Data,name:String) throws -> ChatAttachment? {
        guard let source=CGImageSourceCreateWithData(data as CFData,nil),let type=CGImageSourceGetType(source),UTType(type as String)?.conforms(to:.image)==true else{return nil}
        guard let info=CGImageSourceCopyPropertiesAtIndex(source,0,nil) as? [CFString:Any],
              let width=info[kCGImagePropertyPixelWidth] as? Int, let height=info[kCGImagePropertyPixelHeight] as? Int,
              width>0,height>0,width<=40000,height<=40000,width*height<=40_000_000 else {throw failure("Immagine troppo grande o non leggibile: massimo 40 megapixel.")}
        let options:[CFString:Any]=[kCGImageSourceCreateThumbnailFromImageAlways:true,kCGImageSourceCreateThumbnailWithTransform:true,kCGImageSourceThumbnailMaxPixelSize:2048,kCGImageSourceShouldCacheImmediately:true]
        guard let image=CGImageSourceCreateThumbnailAtIndex(source,0,options as CFDictionary) else{throw failure("Immagine non decodificabile.")}
        let output=NSMutableData()
        guard let destination=CGImageDestinationCreateWithData(output,UTType.jpeg.identifier as CFString,1,nil) else{throw failure("Conversione immagine non disponibile.")}
        CGImageDestinationAddImage(destination,image,[kCGImageDestinationLossyCompressionQuality:0.85] as CFDictionary)
        guard CGImageDestinationFinalize(destination),output.length<=maxImageBytes else{throw failure("Immagine oltre 2 MiB dopo la conversione. Riducila prima di allegarla.")}
        return ChatAttachment(name:name,text:"",originalBytes:data.count,imageData:output as Data,imageMIME:"image/jpeg",imageWidth:image.width,imageHeight:image.height)
    }
    static func validateImage(_ file:ChatAttachment)->Bool {
        guard let data=file.imageData,!data.isEmpty,data.count<=maxImageBytes,file.imageMIME=="image/jpeg",
              let source=CGImageSourceCreateWithData(data as CFData,nil),CGImageSourceGetType(source) as String? == UTType.jpeg.identifier,let info=CGImageSourceCopyPropertiesAtIndex(source,0,nil) as? [CFString:Any],
              let width=info[kCGImagePropertyPixelWidth] as? Int,let height=info[kCGImagePropertyPixelHeight] as? Int else{return false}
        return width>0 && height>0 && width<=2048 && height<=2048
    }
}
