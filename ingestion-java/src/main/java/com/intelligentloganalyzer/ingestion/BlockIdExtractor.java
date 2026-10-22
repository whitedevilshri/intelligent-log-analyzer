package com.intelligentloganalyzer.ingestion;

import java.util.LinkedHashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class BlockIdExtractor {

    private static final Pattern BLOCK_ID_PATTERN =
            Pattern.compile("blk_-?\\d+");

    private BlockIdExtractor() {
    }

    public static Set<String> extractDistinct(String content) {
        Set<String> blockIds = new LinkedHashSet<>();

        if (content == null || content.isBlank()) {
            return blockIds;
        }

        Matcher matcher = BLOCK_ID_PATTERN.matcher(content);

        while (matcher.find()) {
            blockIds.add(matcher.group());
        }

        return blockIds;
    }
}