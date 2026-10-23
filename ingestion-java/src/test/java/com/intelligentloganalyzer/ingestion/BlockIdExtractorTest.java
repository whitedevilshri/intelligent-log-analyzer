package com.intelligentloganalyzer.ingestion;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BlockIdExtractorTest {

    @Test
    void returnsEmptySetForNullOrBlankContent() {
        assertTrue(BlockIdExtractor.extractDistinct(null).isEmpty());
        assertTrue(BlockIdExtractor.extractDistinct("").isEmpty());
        assertTrue(BlockIdExtractor.extractDistinct(" \t\r\n ").isEmpty());
    }

    @Test
    void extractsPositiveAndNegativeBlockIds() {
        Set<String> actual = BlockIdExtractor.extractDistinct(
                "allocated blk_123 and received blk_-987"
        );

        assertEquals(List.of("blk_123", "blk_-987"), new ArrayList<>(actual));
    }

    @Test
    void preservesFirstSeenOrderAndRemovesDuplicates() {
        Set<String> actual = BlockIdExtractor.extractDistinct(
                "blk_10 then blk_-20 then blk_10 again then blk_30"
        );

        assertEquals(
                List.of("blk_10", "blk_-20", "blk_30"),
                new ArrayList<>(actual)
        );
    }

    @Test
    void extractsIdsFromTypicalHdfsMessagesWithoutTreatingIpNumbersAsBlockIds() {
        String content = "BLOCK* NameSystem.addStoredBlock: blockMap updated: "
                + "10.251.106.10:50010 is added to blk_-2126554733521224025 size 67108864; "
                + "Received block blk_872694497849122755 from /10.251.106.10";

        assertEquals(
                List.of("blk_-2126554733521224025", "blk_872694497849122755"),
                new ArrayList<>(BlockIdExtractor.extractDistinct(content))
        );
    }

    @Test
    void ignoresMalformedBlockIdLookalikes() {
        String content = "xblk_123 blk_456abc blk__789 blk_ blk_X12 "
                + "valid=blk_321, also (blk_-654).";

        assertEquals(
                List.of("blk_321", "blk_-654"),
                new ArrayList<>(BlockIdExtractor.extractDistinct(content))
        );
    }

    @Test
    void doesNotExtractUnrelatedNumbersOrText() {
        assertTrue(
                BlockIdExtractor.extractDistinct("pid=123 level=50010 block_id missing").isEmpty()
        );
    }
}
