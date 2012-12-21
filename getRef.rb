(File.new("result.dat").each.map{|l|t=l.chomp.split[0].to_i;t==0?nil: t}).compact.tap{|a|p (a.inject :+)/a.size}
